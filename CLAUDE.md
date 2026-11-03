# 5J実験：課題解決型の機械学習を用いたロボット制御

LEGO EV3とRaspberry Piを使い，カメラ画像から印刷された数字をMNISTで学習したモデルで判定し，数字の書かれた的を順番通りに倒す自律ロボットを作るプロジェクト．
Claude Codeは **Raspberry Pi上** で動作している．
`../5j/` は旧版．参考にせず，修正もしない．

## 課題

- ある程度の広さの床に，数字（1，2，3）を印刷した軽い板と障害物がランダムに立てられる．
- 的は必ず 1 → 2 → 3 の順番で倒す．倒し方の指定はない．
- 障害物に触れたり倒したりしても，失敗や減点にはならない．
- 床の広さ，板の大きさ，制限時間は決まっていない．これらの値を前提にしたコードを書くときは，パラメータとして変更できるようにする．
- 数字の検出はRaspberry Piに繋いだカメラ（Logicool C270N）で行う．
- 制御プログラム実行前のロボットのサイズは 300mm × 300mm × 300mm とする．
- Raspberry Piはモバイルバッテリーでロボットに載せる．実走時はPCの操作なしで動くようにする．
- 作成したソースコードは全て提出する（テキストファイル `.txt` 形式）．

## システム構成

```
PC ──(Tailscale + SSH)──> Raspberry Pi ──(有線LAN / TCP)──> EV3
                          TCPクライアント                   TCPサーバ（ポート9999）
                          ・カメラ画像取得（OpenCV）
                          ・数字判定（TensorFlow / Keras）
                          ・制御指令の決定
                                                            ・モータ / センサ制御のみ
```

| 項目 | Raspberry Pi | EV3 |
|---|---|---|
| 役割 | 画像処理・推論・判断 | ハードウェア制御のみ |
| TCP | クライアント（EV3へ接続する） | サーバ（接続を待ち受ける） |
| 言語 / 実行環境 | CPython 3.11以上（uvで管理） | pybricks-micropython（MicroPython） |
| コードの場所 | `raspi_client/` | `ev3_server/` |
| 編集場所 | ここ（Raspberry Pi） | ここ（Raspberry Pi）で書いて転送 |

### 機能の切り分け（最重要）

- **推論・画像処理・判断ロジックはすべてRaspberry Pi側に書く．**
- EV3側は「Raspberry Piから受け取った指令でモータを動かす」「センサ値を返す」だけにする．
- EV3側のコードで numpy / tensorflow / opencv などをimportしない．EV3では動かない．
- 両者の通信仕様（メッセージ形式）を変更するときは，`raspi_client/` と `ev3_server/` の両方を必ず同時に修正する．

### 通信形式

- EV3 → Raspberry Pi：`distance_mm,gyro_angle\n`（超音波センサの距離(mm)，ジャイロの角度(度)）
- Raspberry Pi → EV3：`left_speed,right_speed\n`（左右のモータ速度．整数，単位は度/秒）
- EV3がセンサ値を1行送るたびに，Raspberry Piが速度指令を1行返す．
- 指令が1秒届かないと，EV3はモータを停止する．
- 形式を変更するときは，ユーザに確認してからここも修正する．

## Raspberry Pi側（`raspi_client/`）

- 依存関係は uv で管理している（`pyproject.toml`, `uv.lock`）．
  - 実行：`cd raspi_client && uv run python <ファイル名>`
  - 追加：`uv add <パッケージ>`．`pip install` は使わない．
- 主なファイル
  - `camera.py`：カメラからのフレーム取得と認識エリアの計算（`LatestFrameReader`, `center_roi`）
  - `digit_recognizer.py`：モデル読み込み・前処理・推論（`load_model`, `preprocess_image`, `predict_digit`）
  - `robot_controller.py`：TCPクライアント本体．カメラ → 推論 → EV3へ速度指令を送信．
  - `recognition_viewer.py`：EV3なしで認識だけを試すツール（`--web` でブラウザ配信）
- 学習済みモデル `mnist_model.keras` が存在しない場合は，勝手に再学習せず，ユーザに確認する．
- MNIST程度の学習はRaspberry Pi上で行ってよい．ただし学習の実行前にはユーザに確認する（実行中は推論・カメラ処理が重くなるため）．

## EV3側（`ev3_server/`）

- 主なファイル
  - `motor_sensor_server.py`：TCPサーバ本体（ポート9999）
- 実行環境は **pybricks-micropython**（ev3dev上）．ev3dev2ライブラリではない．
  - 先頭行：`#!/usr/bin/env pybricks-micropython`
  - API：`pybricks.hubs`, `pybricks.ev3devices`, `pybricks.parameters`, `pybricks.tools`
- MicroPythonの制約に注意する．
  - f-stringは使わない（文字列連結か `"{}".format()` を使う）．
  - 標準ライブラリは限定的．使えるか不明なモジュールは使わない．
  - `time.sleep` より `pybricks.tools.wait`（ミリ秒）を優先する．
- ソケットを使う場合は `SO_REUSEADDR` を設定する（強制終了後の「使用中」エラー対策）．
- 通信が途絶えたら必ずモータを停止する処理を入れる（`finally` で `motor.brake()`）．

## EV3への転送と実行

- **EV3上のファイルを直接編集しない．** 正本は常にこのリポジトリ．
- 実行するときにファイルを転送する．
- EV3への接続情報：`robot@10.42.0.175`（パスワード認証，SSHエイリアスなし）
  - IPアドレスが変わった場合は，EV3のディスプレイの表示を確認する．
- **Claude Codeはパスワードを入力できない．** EV3に接続するコマンドは自分で実行せず，ユーザに提示して実行してもらう．
- パスワードをファイルやコマンドに書かない．
- コマンド（`2_5j/` で実行．ファイル名と時間は必要に応じて変える）：
  ```
  scp ev3_server/motor_sensor_server.py robot@10.42.0.175:~/
  ssh -t robot@10.42.0.175 'timeout 120 brickrun -r -- pybricks-micropython motor_sensor_server.py'
  ```
- 起動順はEV3（サーバ）が先，Raspberry Pi（クライアント）が後．
- EV3は非常に低速．SSH接続やpybricksの起動に数秒〜十数秒かかるのは正常．すぐにリトライしない．

## 安全ルール

- **モータが動く可能性のある実行は，必ず事前にユーザへ確認する．**（ロボットの落下・暴走防止）
- EV3で実行するコマンドには，EV3側で動くように `timeout` を付ける．終了後はモータの停止を確認する．
- 緊急停止（プログラムを終了して全モータ停止．ユーザが実行する）：
  ```
  ssh robot@10.42.0.175 'pkill -f "[p]ybricks-micropython"; for m in /sys/class/tacho-motor/motor*; do echo reset > $m/command; done'
  ```
- EV3上で `pip install` / `apt install` を実行しない．必要な場合はユーザに相談する．

## コーディング規約

- コメント・ログ出力は日本語でよい．ログの先頭に `[RasPi]` / `[EV3]` を付けて出力元を区別する．
- ソースコードはグループで提出するため，コメントはできるだけ丁寧に書く．
- パラメータ（しきい値・ポート番号・IPアドレスなど）はファイル先頭に定数としてまとめる．
- 文章（ドキュメント・コメント）の句読点は「．」「，」を使う．
