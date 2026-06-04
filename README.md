# 🌽 Maize Breeding Analysis System
**NSFCRC Drought Tolerance Breeding Program**

## วิธีติดตั้งและรันโปรแกรม

### ขั้นที่ 1 — ติดตั้ง Python dependencies
```bash
pip install -r requirements.txt
```

### ขั้นที่ 2 — รันโปรแกรม
```bash
streamlit run app.py
```
เปิด browser ไปที่ `http://localhost:8501`

---

## ไฟล์ที่ต้อง Upload

| ไฟล์ | Upload ช่อง | ตัวอย่าง |
|---|---|---|
| Nursery | Nursery (P……xlsx) | P226101.xlsx |
| Yield Trial WS | Yield Trial WS | YT226101WS.xlsx |
| Yield Trial WW | Yield Trial WW | YT226101WW.xlsx |
| Yield Trial อื่นๆ | Yield Trial อื่นๆ | YT226102.xlsx |

## Modules

1. **🌿 Nursery Analysis** — ดูข้อมูล selfing, คัดเลือกสายพันธุ์
2. **📊 Yield Trial Analysis** — Adjusted means, ranking vs checks
3. **💧 Drought Tolerance** — DTI, SSI, WS vs WW
4. **📈 Multi-Trial Summary** — ภาพรวมทุก trial
