# Dual-WaterMark FastAPI 文件

本文件說明 `Dual-WaterMark/api.py` 提供的 HTTP API。

## 1) 啟動 API

在專案根目錄執行：

```bash
cd /home/project/Documents/Dual-WaterMark
./run_api.sh 0.0.0.0 7867
```

- 預設 Python：`/home/project/Documents/EditGuard/envs/editguard/bin/python`
- 服務位址（本機）：`http://127.0.0.1:7867`
- 服務位址：`https://api.watermark.nyanfox.com/`

---

## 2) 快速檢查

### Health Check

```bash
curl -X GET http://127.0.0.1:7867/health
```

回傳：

```json
{"ok": true}
```

### 產生 64-bit 隨機 bits

```bash
curl -X GET http://127.0.0.1:7867/random-bits
```

回傳：

```json
{"bits": "0101..."}
```

---

## 3) API 端點總覽

### `POST /embed`

用途：執行主流程嵌入（StegaStamp -> EditGuard）。

#### Request JSON

- `image_base64` (string, 必填): 原圖（PNG/JPG 的 base64）
- `editguard_bits` (string, 必填): 64-bit 0/1 字串
- `stegastamp_secret` (string, 必填): StegaStamp 文字（建議 <= 7 字元）
- `editguard_root` (string, 選填)
- `stegastamp_root` (string, 選填)
- `stegastamp_env` (string, 選填)
- `stegastamp_model_path` (string, 選填)

#### Response JSON

- `metadata_json` (string)
- `stegastamp_image_base64` (string)
- `final_image_base64` (string)
- `stegastamp_residual_base64` (string)

---

### `POST /inpaint`

用途：根據使用者提供 mask 與 prompt 做 EditGuard inpaint。

#### Request JSON

- `image_base64` (string, 必填): 要修補的圖
- `mask_base64` (string, 必填): 遮罩圖（白色區域為要修補）
- `prompt` (string, 選填): 預設 `repair tampered region`
- `editguard_root` (string, 選填)
- `stegastamp_root` (string, 選填)
- `stegastamp_env` (string, 選填)

#### Response JSON

- `inpainted_image_base64` (string)

---

### `POST /verify`

用途：驗證圖片（StegaStamp decode + EditGuard reveal）。

#### Request JSON

- `image_base64` (string, 必填)
- `metadata_json` (string, 必填): 由 `/embed` 回傳
- `editguard_root` (string, 選填)
- `stegastamp_root` (string, 選填)
- `stegastamp_env` (string, 選填)
- `stegastamp_model_path` (string, 選填)

#### Response JSON

- `stegastamp_found_codes` (array)
- `editguard_recovered_bits` (string)
- `editguard_accuracy` (string)
- `editguard_mask_base64` (string)
- `summary` (object)

---

## 4) curl 呼叫範例

下面示範 `embed -> verify` 基本流程。

### Step A: 把圖片轉成 base64

```bash
IMG_B64=$(base64 -w 0 /home/project/Documents/ttttt.png)
```

### Step B: 呼叫 `/embed`

```bash
curl -s -X POST http://127.0.0.1:7867/embed \
  -H "Content-Type: application/json" \
  -d "{\
    \"image_base64\": \"$IMG_B64\",\
    \"editguard_bits\": \"0101010101010101010101010101010101010101010101010101010101010101\",\
    \"stegastamp_secret\": \"Stega!\"\
  }" > /tmp/embed_resp.json
```

### Step C: 取出嵌入後圖片與 metadata，再呼叫 `/verify`

```bash
FINAL_B64=$(jq -r '.final_image_base64' /tmp/embed_resp.json)
META_JSON=$(jq -r '.metadata_json' /tmp/embed_resp.json)

curl -s -X POST http://127.0.0.1:7867/verify \
  -H "Content-Type: application/json" \
  -d "{\
    \"image_base64\": \"$FINAL_B64\",\
    \"metadata_json\": $(jq -Rs . <<< "$META_JSON")\
  }"
```

---

## 5) Python client 範例

專案已提供範例：`api_client_example.py`

### 安裝依賴

```bash
pip install requests
```

### 執行（embed -> verify）

```bash
cd /home/project/Documents/Dual-WaterMark
python api_client_example.py \
  --base-url http://127.0.0.1:7867 \
  --image /home/project/Documents/ttttt.png
```

### 執行（embed -> inpaint -> verify）

```bash
cd /home/project/Documents/Dual-WaterMark
python api_client_example.py \
  --base-url http://127.0.0.1:7867 \
  --image /home/project/Documents/ttttt.png \
  --mask /path/to/your_mask.png \
  --prompt "repair tampered region"
```

輸出會存到 `--out-dir`（預設 `./output_api_client`）。

---

## 6) base64 格式說明

API 支援兩種格式：

1. 純 base64（推薦）
2. Data URL（例如 `data:image/png;base64,...`）

---

## 7) 常見錯誤

- `Invalid base64 image input.`
  - 代表 `image_base64` 不是合法 base64。
- `Mask is required for EditGuard inpaint.`
  - `/inpaint` 沒有給 `mask_base64`。
- checkpoint 或路徑相關錯誤
  - 請檢查 `EditGuard` 與 `StegaStamp-pytorch` 路徑與模型檔。

---

## 8) 版本

- API 程式：`api.py`
- 目前版本：`1.0.0`
