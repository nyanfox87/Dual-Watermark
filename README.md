# Dual-WaterMark

Dual-WaterMark 是一個兩頁式 Gradio Demo，整合 **EditGuard** 與 **StegaStamp-pytorch**：

1. **Add Watermark Pipeline**
   - 先嵌入 EditGuard（64-bit）
   - 再嵌入 StegaStamp（<= 7 UTF-8 bytes）
   - 回傳最終圖片與 `metadata JSON`（供檢驗）

2. **Detect & Verify**
   - 不需移除浮水印，直接輸入含浮水印圖片
  - 輸出 EditGuard 防竄改可視化遮罩與 StegaStamp 解碼文字
   - 貼上第一頁 metadata JSON 進行一致性驗證

## 目錄

- `app.py`: 主 Gradio 入口
- `api.py`: FastAPI 入口
- `API.md`: FastAPI 使用文件
- `api_client_example.py`: FastAPI Python 呼叫範例
- `adapters/editguard_adapter.py`: EditGuard 封裝
- `adapters/stegastamp_adapter.py`: StegaStamp 封裝
- `services/pipeline.py`: 第一頁嵌入 pipeline
- `services/verify.py`: 第二頁驗證流程
- `schemas/metadata.py`: metadata JSON schema 與驗證

## 前置需求

- EditGuard 專案：`/home/project/Documents/EditGuard`
  - 需要 `checkpoints/clean.pth`
  - 需要可執行 Python（預設：`/home/project/Documents/EditGuard/envs/editguard/bin/python`）
- StegaStamp-pytorch 專案：`/home/project/Documents/StegaStamp-pytorch`
  - 需要 checkpoint：`asset/best.pth`
  - 若是 Git LFS pointer 檔（小文字檔），請先在該目錄執行 `git lfs install && git lfs pull`

## 啟動

```bash
cd /home/project/Documents/Dual-WaterMark
chmod +x run.sh
./run.sh
```

預設會在 `0.0.0.0:7868` 啟動。

## metadata JSON

第一頁會產出以下欄位：

- `version`
- `created_at`
- `editguard_bits_expected`
- `stegastamp_secret_expected`
- `final_image_sha256`

第二頁會使用這些欄位驗證：

- `editguard_intact`
- `copyright_match`
- `fingerprint_match`
- `overall_pass`

## 備註

- 主程式在 EditGuard 環境運行，StegaStamp 直接透過 `StegaStamp-pytorch` 進行 PyTorch 推論（不再透過 TensorFlow 子程序）。
- EditGuard bits 固定為 64-bit；StegaStamp secret 固定最多 7 字元（含 BCH 錯誤更正）。
