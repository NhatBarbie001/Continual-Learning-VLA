# Pipeline SmolVLA trên LIBERO — đọc từ code

*Nguồn: đọc ở LeRobot commit `f5c9bc0`. **Commit đã pin trên Kaggle: `ba83e4cf43737e2665be0aafcb5f513b83c7d0f3`.** Mọi trích dẫn `file:dòng` tính từ `src/lerobot/`.*
*Đã so hai commit: phần SmolVLA, flow matching, LIBERO env và env processor chỉ khác 1 chỗ: cách cắt KV cache trong `denoise_step` (`modeling_smolvla.py:856-858`). Thay đổi này không ảnh hưởng nội dung bên dưới; các dòng sau 857 lệch +1.*

**Các file đã đọc:**
- `policies/smolvla/configuration_smolvla.py`
- `policies/smolvla/processor_smolvla.py`
- `policies/smolvla/modeling_smolvla.py`
- `policies/smolvla/smolvlm_with_expert.py`
- `policies/common/flow_matching.py`
- `policies/common/vla_utils.py`
- `envs/libero.py`
- `envs/configs.py`
- `processor/env_processor.py`

---

## Tóm tắt một hình

```
LIBERO env ──► env processor ──► policy preprocessor ──► VLAFlowMatching ──► postprocessor ──► env.step
(2 camera,     (lật ảnh 180°,     (đổi tên key, thêm      prefix: [ảnh₁ 64 tok][ảnh₂ 64 tok]     (bỏ chuẩn hoá
 state thô)     state 8 chiều)     "\n", tokenize ≤48,             [lời dẫn ≤48 tok][state 1 tok]  action)
                                   chuẩn hoá MEAN_STD)    suffix: [50 token action có nhiễu + t]
                                                          → 10 bước Euler → chunk 50×7
                                                          → hàng đợi n_action_steps
```

## ① Inputs

| Thành phần | Giá trị trong code | Trích dẫn |
|---|---|---|
| Camera | `agentview_image` → `image`, `robot0_eye_in_hand_image` → `image2` | `envs/libero.py:116, 156-160` |
| Kích thước render khi eval | **360×360** (mặc định của `LiberoEnv` config) | `envs/configs.py:357-358` |
| Kích thước ảnh trong dataset train | 256×256 | LeRobot LIBERO docs |
| Lật ảnh | lật 180° (cả H và W) để khớp quy ước của dataset | `processor/env_processor.py:59` |
| State | 8 chiều = vị trí eef (3) + quaternion đổi sang axis-angle (3) + vị trí 2 ngón gripper (2) | `processor/env_processor.py:68-73` |
| Instruction | thêm ký tự xuống dòng, tokenize bằng tokenizer của SmolVLM2, tối đa **48 token**, pad bên phải theo câu dài nhất trong batch | `policies/smolvla/processor_smolvla.py:68-74`; config `:60, :94` |
| Chuẩn hoá | ảnh: giữ nguyên; state và action: **MEAN_STD** theo stats của dataset | `configuration_smolvla.py:32-38`; `processor_smolvla.py:76` |

**Xử lý ảnh bên trong policy** (`modeling_smolvla.py:385-433`):
1. Đổi uint8 sang khoảng [0, 1].
2. **Resize giữ tỉ lệ rồi pad lên 512×512** (pad bên trái và trên, giá trị 0).
3. Đổi sang khoảng [−1, 1] theo yêu cầu của SigLIP.
4. Camera nào thiếu thì thay bằng ảnh rỗng có mask = 0, nhưng chỉ khi config cho phép `empty_cameras`.

**State bên trong policy:** pad từ 8 lên **32 chiều** (`modeling_smolvla.py:462-466`, config `:41`).

## ② Components

### VLM (SmolVLM2-500M; biến thể và số layer phụ thuộc checkpoint, xem bảng config bên dưới)
- **Giữ `num_vlm_layers` layer đầu** của text model nếu giá trị > 0 (`smolvlm_with_expert.py:102-104`). Mặc định và `smolvla_base` là 16; riêng `smolvla_libero` là 0, tức không cắt.
- **Đóng băng toàn bộ VLM** khi `train_expert_only=True`, bao gồm cả vision encoder (`smolvlm_with_expert.py:152-160`, config `:69-71`).
- Trọng số VLM được nạp ở **bfloat16** (`smolvlm_with_expert.py:92-96`). ⚠️ Xem rủi ro với T4 ở cuối file.
- **Ảnh:** SigLIP, rồi connector (pixel shuffle + chiếu chiều), rồi nhân √d (`modeling_smolvla.py:632-637`, `smolvlm_with_expert.py:202-215`).
  - Số token mỗi ảnh theo tính toán là (512/16)² / 4² = **64**.
  - Cần in `img_emb.shape` khi chạy để xác nhận.
- **Lời dẫn:** embedding của VLM, nhân √d (`modeling_smolvla.py:660-663`).
- **State: là một token trong prefix, đi qua VLM**, chứ không đi qua action expert. `state_proj: Linear(32 → hidden của VLM)` (`modeling_smolvla.py:561-563, 671-673`).

### Prefix và mask
- Thứ tự prefix: `[ảnh 1][ảnh 2][lời dẫn][state]`.
- Ảnh và lời dẫn nhìn được lẫn nhau theo hai chiều.
- **State mở một khối mới**, nên ảnh và lời dẫn không nhìn thấy state (`modeling_smolvla.py:681-682`).

### Action expert
- Cùng kiến trúc với text model của SmolVLM2, **số layer bằng số layer VLM giữ lại** (`smolvla_base`: 16).
- Rộng **`expert_width_multiplier`×** (`smolvla_base`: 0.75; `smolvla_libero`: 0.5), intermediate size tính lại theo độ rộng mới (`smolvlm_with_expert.py:108-113`, config `:96-99`).
- Expert có embedding riêng; `embed_tokens` bị xoá (`:138`).

**Đầu vào của expert (suffix)** (`modeling_smolvla.py:698-739`):
- Mỗi token action có nhiễu đi qua `action_in_proj` (32 → hidden của expert).
- Ghép với **time embedding dạng sin-cos** (period 4e-3 … 4.0).
- Qua MLP: `Linear → SiLU → Linear`.
- Kết quả là **50 token**.

**Cách expert gắn với VLM** (`smolvlm_with_expert.py:448-476`):

| Layer | Loại | Expert nhìn thấy gì | Trích dẫn |
|---|---|---|---|
| Chẵn (0, 2, …, 14) | **Self-attention chung**: Q/K/V của prefix và suffix được nối lại, attention theo mask | prefix + các token action **đứng trước và chính nó** | `smolvlm_with_expert.py:220-292` |
| Lẻ (1, 3, …, 15) | **Cross-attention**: Q từ expert; K/V là K/V của VLM ở layer đó, chiếu qua `k_proj`/`v_proj` riêng của expert | **chỉ prefix**, không nhìn các token action khác | `:122-136, 294-398` |

- **Mask của suffix là `[1]*50`**, nên các token action **causal với nhau**: token thứ i chỉ nhìn các token 1…i (`modeling_smolvla.py:733-734`; ý nghĩa của mask xem `vla_utils.py:60-80`).
- **Tham số được train:** action expert, `action_in_proj`, `action_out_proj`, MLP thời gian, và `state_proj` (vì `train_state_proj=True`).

## ③ Loss (training)

Trong `modeling_smolvla.py:741-774` và `:293-354`:

1. **Lấy mẫu:**
   - nhiễu ε ~ N(0, I) cùng shape với action đã pad (B × 50 × 32);
   - thời gian t = Beta(1.5, 1) · 0.999 + 0.001 (`:601-602`).
2. **Tạo input và target** (quy ước `NOISE_AT_ONE`, `flow_matching.py:247-248`):
   - x_t = t·ε + (1 − t)·A
   - u_t = ε − A
3. **Dự đoán:** v_t = `action_out_proj`(50 token cuối của expert), tính ở fp32 (`modeling_smolvla.py:769-772`).
4. **Loss từng phần tử:** `mse(u_t, v_t, reduction="none")` (`:773`).
5. **Cắt về 7 chiều thật của action** (`:325`). Nhân với mask `~action_is_pad` để bỏ các bước vượt quá cuối episode (`:328-331`).
6. **Lấy trung bình** trên các phần tử hợp lệ (`:351-352`).

**Optimizer mặc định:** AdamW lr 1e-4, betas (0.9, 0.95), clip grad 10; warmup 1000 step, cosine decay về 2.5e-6 trong 30k step (`configuration_smolvla.py:74-83`).

## ④ Action generation (inference)

Trong `modeling_smolvla.py:776-862`:

1. **Chạy prefix qua VLM một lần**, lưu K/V vào `DynamicCache` (`:796-808`).
2. **Bắt đầu từ** x = ε ~ N(0, I), shape B × 50 × 32.
3. **Tích phân Euler `num_steps = 10` bước**, từ t = 1 về 0 với dt = −1/10 (`flow_matching.py:338`; config `:63`).
   - Mỗi bước chỉ chạy expert trên suffix và đọc K/V của prefix từ cache.
   - Sau mỗi bước, cắt cache về đúng độ dài prefix (`:827-862`).
4. **Cắt về 7 chiều** (`:226`), rồi postprocessor bỏ chuẩn hoá (`processor_smolvla.py:78-81`).
5. **Hàng đợi action** (`:256-288`):
   - Đưa `n_action_steps` action đầu của chunk vào queue.
   - Mỗi bước env lấy ra một action.
   - **Hết queue mới gọi lại model.**
   - Mặc định `n_action_steps = 50 = chunk_size` (config `:29-30`).
6. **Action** là 7 chiều: 6 chiều delta eef + 1 chiều gripper, trong khoảng [−1, 1]. Env chạy ở chế độ `relative` (delta) (`envs/libero.py:129, 354-361`).

## ⑤ Eval metric

| Thành phần | Giá trị | Trích dẫn |
|---|---|---|
| Khởi tạo episode | reset, đặt **init state cố định** của LIBERO (mỗi episode một init state khác), rồi chạy **10 bước đứng yên** để vật thể ổn định | `envs/libero.py:344-352` |
| Kết thúc episode | `check_success()` của LIBERO trả về True, **hoặc** chạm số bước tối đa | `envs/libero.py:376-377, 189-190` |
| Số bước tối đa | Spatial 280 · Object 280 · Goal 300 · Long (`libero_10`) 520 | `envs/libero.py:98-104` |
| Metric | **Success rate** = % episode có `is_success` | `scripts/lerobot_eval.py:336-356` |
| Mặc định của eval | hard reset, `init_states=True` | `envs/configs.py:354-355` |

---

## Config thực tế của các checkpoint (đọc `config.json` trên Hub, 03/11)

| Thông số | Paper SmolVLA | `lerobot/smolvla_base` (dùng cho Phase 4) | `HuggingFaceVLA/smolvla_libero` (dùng cho R1) |
|---|---|---|---|
| VLM | SmolVLM2-500M | `SmolVLM2-500M-Video-Instruct` | **`SmolVLM2-500M-Instruct`** |
| Số layer VLM | 16 (cắt nửa) | 16 | **`num_vlm_layers = 0` → giữ nguyên toàn bộ layer** (code chỉ cắt khi > 0, `smolvlm_with_expert.py:102`) |
| Độ rộng expert | 0.75× | 0.75 | **0.5** |
| Self-attention mỗi n layer | 2 | 2 | 2 |
| `n_action_steps` | không nêu cho LIBERO | 50 | **1** (dự đoán lại ở mỗi bước) |
| Camera | — | `camera1/2/3`, 256×256 | `image`, `image2`, 256×256 |
| State / action | — | 6 / 6 chiều | 8 / 7 chiều |
| `num_steps` Euler | 10 | 10 | 10 |

**Hệ quả:**
1. **Checkpoint LIBERO công khai không có cùng kiến trúc với SmolVLA trong paper**: VLM khác, không cắt layer, expert hẹp hơn. Đây là khác biệt lớn nhất cần ghi vào bảng differences. Nó cũng có thể giải thích một phần vì sao số đo lại (khoảng 76%) thấp hơn paper (87.3%).
2. **R1 dùng mặc định `n_action_steps = 1`**, gần cấu hình tốt nhất trong Table 13 của paper (1 → 80.3; 10 → 82.8 là tốt nhất; 50 → 51.8). Không cần chọn lại, nhưng eval sẽ tốn hơn: mỗi bước env là một lần chạy cả VLM + 10 bước Euler. Cần đo t_ep ở S0.
3. **Phase 4 dùng `smolvla_base`**, đúng kiến trúc của paper (16 layer, 0.75). Muốn fine-tune trên LIBERO cần đổi tên camera:
   `--rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'`
   (flag có trong `configs/train.py:201`). Camera 3 bị thiếu; vì `empty_cameras = 0` nên được bỏ qua (`modeling_smolvla.py:426-428`).
   State 8 chiều và action 7 chiều đều pad lên 32 nên dùng được với `state_proj` và `action_in_proj` cũ, nhưng stats chuẩn hoá phải lấy từ dataset LIBERO.
4. **Video eval:** `lerobot-eval` tự render **10 episode đầu của mỗi task** vào video (`scripts/lerobot_eval.py:832`). Với 10 ep/task thì mọi episode đều có video, đủ cho Phase 3.

## Khác với những gì đã ghi trong plan và slide (đã sửa)

1. **State đi vào prefix của VLM, không đi vào action expert.**
   - Plan cũ ghi "input expert = noisy chunk + timestep + state token" là sai.
   - Đã sửa trong plan Phase 2 và slide 10.
2. **Các layer xen kẽ không đối xứng.**
   - Ở layer chẵn, expert nhìn cả prefix và các token action trước nó.
   - Ở layer lẻ, expert chỉ cross-attend vào K/V của VLM.
   - Giữa các token action là attention **causal**, khác π0: ở π0 các token action nhìn nhau hai chiều.
3. **Ảnh khi eval render ở 360×360, còn dataset train là 256×256.** Cả hai sau đó đều được resize-pad lên 512 nên vẫn chạy được. Tuy vậy đây là một điểm lệch phân phối giữa train và eval cần ghi vào bảng differences.
   - R3 nên thử thêm `--env.observation_height=256 --env.observation_width=256`.

## Rủi ro cho Kaggle (thêm vào checklist của S0)

- ⚠️ **bfloat16 trên T4:** VLM được nạp ở bf16 (`smolvlm_with_expert.py:94`), nhưng T4 (sm_75) không có tensor core cho bf16.
  - Có thể chạy chậm, hoặc báo lỗi ở một số kernel.
  - Nếu S0 lỗi hoặc rất chậm, cần ép VLM sang fp16 hoặc fp32 sau khi nạp, và ghi lại thay đổi này.
- ⚑ **Còn cần xác nhận khi chạy:**
  - số visual token mỗi ảnh (64);
  - số layer VLM thực tế của `smolvla_libero` (theo config là giữ toàn bộ text model của SmolVLM2-500M-Instruct);
  - thời gian 1 episode khi `n_action_steps = 1`.
- ✅ Đã xác nhận từ `config.json`: `n_action_steps = 1`, `resize_imgs_with_padding = 512×512`, `empty_cameras = 0`.
