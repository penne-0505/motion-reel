# motion-reel — 15 秒モーションデザイン・ショーリール

「モーションデザイナーとしての実力を示す 15 秒のショーリールを全力で」という依頼を受けて、Claude（Opus 5.5）が Claude Code 上で制作したもの（2026-09-25）。GUI のアニメーションソフトは使っておらず、映像・3D・音のすべてをコードから生成している。

**動画**: [Releases](https://github.com/penne-0505/motion-reel/releases/latest) から
- `motion_reel.mp4` — マスター（1920×1080 / 60fps / H.264 CRF14 / AAC 320k、約 118MB）
- `motion_reel_share.mp4` — 共有用（CRF21・grain tune、約 57MB）

## 構成（120 BPM、1 拍 = 0.5 秒 = 30 フレーム、全 30 拍）
| 拍 | 秒 | 場面 |
|---|---|---|
| 0–2 | 0–1 | 点 → スクワッシュ → 線に伸びて紙色のスレートへ |
| 2–8 | 1–4 | MAKE / IT（可変フォントの太さ 100→900）/ MOVE → エコースタック → O のカウンターへ潜るズーム |
| 8–14 | 4–7 | 波紋グリッド → 市松ひし形 → 点を追う流れ場 → ストライプで画面を塗る |
| 14–20 | 7–10 | Blender 3D：落下と着地のスクワッシュ、一斉ジャンプ、ウェーブ、ホイップパン |
| 20–24 | 10–12 | パーティクル：流入 → 銀河状の渦 → ハーフトーンの「動」（重＝紙色／力＝オレンジ）→ 爆発 |
| 24–30 | 12–15 | エンドカード。冒頭の点が句点として落ちてくる（ブックエンド） |

## ファイル
- `reel.py` — 2D シーン、合成、ポストエフェクト。各フレームを「拍 → 絵」の関数として描く
- `scene3d.py` — Blender 用スクリプト。全トランスフォームを拍から計算し、毎フレームにキーを打つ
- `audio.py` — サウンドトラック。全ての音を numpy で合成し、映像と同じ拍の格子に置く

## 必要なもの
- Python 3.13 と `requirements.txt` のパッケージ
- Blender 5.1（flatpak 版 `org.blender.Blender` で確認）
- ffmpeg
- フォント: Adwaita Sans / Adwaita Mono（`adwaita-fonts`）、Noto Sans CJK Black（`noto-fonts-cjk`）。パスは `reel.py` の `load_fonts()` で Arch Linux の配置に決め打ちしている

## 再生成
```
uv venv .venv && uv pip install --python .venv -r requirements.txt
.venv/bin/python audio.py                     # soundtrack.wav
flatpak run --filesystem=$PWD org.blender.Blender -b -P $PWD/scene3d.py   # render3d/（約 3.5 分）
.venv/bin/python reel.py                      # frames/（約 4 分、範囲指定: reel.py 600-900）
ffmpeg -framerate 60 -i frames/%04d.png -i soundtrack.wav -c:v libx264 -crf 14 -pix_fmt yuv420p -c:a aac -b:a 320k -shortest motion_reel.mp4
```
- 2D は pycairo + fontTools（Adwaita Sans の可変軸を直接読む）+ numpy/OpenCV。モーションブラーはサブフレーム蓄積
- 3D は Blender 5.1 Eevee
- 音は F マイナー、-14.3 LUFS / ピーク -0.9 dBFS

## 既知の制約
- 音は波形・スペクトログラム・ラウドネス測定でだけ検証しており、制作者（Claude）は耳で聴いていない
- 文字組みにカーニング（GPOS）を適用していない
- タイミングの数値が `reel.py` / `scene3d.py` / `audio.py` に重複している（共通のキューシートにはなっていない）

## License
MIT
