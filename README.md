# JapaneseCorpus

[![CI](https://github.com/KazumaProject/JapaneseCorpus/actions/workflows/ci.yml/badge.svg)](https://github.com/KazumaProject/JapaneseCorpus/actions/workflows/ci.yml)
[![Corpus release](https://img.shields.io/github/v/release/KazumaProject/JapaneseCorpus?label=corpus)](https://github.com/KazumaProject/JapaneseCorpus/releases/latest)

日本語 Wikipedia と青空文庫から、再現可能な日本語文書コーパスを生成する
パイプラインです。将来のかな漢字変換用言語モデル・辞書作成を目的として、
文書境界、出典、版情報、青空文庫のルビを保持します。

## Download

常に最新の公開版1件だけを
[GitHub Releases](https://github.com/KazumaProject/JapaneseCorpus/releases/latest)
で配布します。

- `wikipedia-<dump-date>-<shard>.jsonl.zst`: Wikipediaの分割コーパス
- `aozora-<commit>.jsonl.zst`: 青空文庫公式リポジトリ由来のコーパス
- `manifest.json`: ソース版、件数、サイズ、警告、生成commit
- `SHA256SUMS`: コーパス資産のSHA-256
- `corpus-record.schema.json`: 1行ごとのJSON Schema
- `mozc-unigram-<part>.txt.zst`: Mozc 5列形式の単語辞書
- `mozc-bigram-<part>.txt.zst`: 隣接2語を句として格納したMozc 5列辞書
- `mozc-trigram-<part>.txt.zst`: 隣接3語を句として格納したMozc 5列辞書
- `mozc-id.def`: 辞書の左右文脈IDに対応するMozc定義
- `ngram-manifest.json` / `NGRAM-SHA256SUMS`: 辞書の件数、生成条件、検証値
- `mozc-english-reading-unigram-<part>.txt.zst`: 発音が読み全体と一致する、ひらがな読みから英語への直接辞書
- `english-dictionary-manifest.json` / `ENGLISH-DICTIONARY-SHA256SUMS`: 日英辞書の出典、件数、検証値
- `JMdict_e-<date>.xml.gz` / `JMDICT-LICENSE.html`: 使用したJMdict原本とライセンス
- `ajimee-bench-report.json`: AJIMEE-BenchによるAccuracy@1／MinCER評価

展開例:

```console
zstd -dc wikipedia-20260719-00000.jsonl.zst | head -n 1 | jq
```

## Record format

1行が1文書のUTF-8 JSONです。Wikipediaは1ページ、青空文庫は1作品を1文書
とします。

```json
{
  "schema_version": 1,
  "source": "wikipedia",
  "document_id": "12345",
  "title": "記事名",
  "url": "https://ja.wikipedia.org/?curid=12345",
  "text": "正規化済み本文",
  "metadata": {
    "dump_date": "20260719",
    "timestamp": "2026-07-18T00:00:00Z"
  }
}
```

青空文庫レコードは `metadata` に人物・役割・文字遣いを持ちます。ルビを
取得できた場合は `annotations.ruby` に、本文のUnicodeコードポイント単位の
`[start, end)` と読みを保存します。

## Data policy

- Wikipedia: 完了済みの最新CirrusSearch `jawiki_content` ダンプから名前空間0を取得
- 青空文庫: 公式拡充版CSVで作品・全関係人物の著作権フラグが `なし` の作品だけを取得
- 共通正規化: UTF-8、LF、Unicode NFC、制御文字・行末空白・過剰空行の除去
- 形態素解析、文分割、表記の現代化、他ソース間の重複除去は行わない
- 日英辞書: 英語版JMdictのカタカナ読みをひらがなへ正規化し、英語訳を候補化

青空文庫の旧字旧仮名作品も、`metadata.orthography` で判別できる形で保持します。
かな漢字変換モデルを作る段階で用途に合わせて選別してください。

## Mozc-format dictionaries

辞書は公開する全コーパスをVibrato/IPADICで形態素解析して生成します。文書・文の
境界を越えてn-gramを作らず、Akazaと同様に頻度16以上の語彙を先に確定します。
unigramは頻度16以上、bigram/trigramは頻度32以上を収録します。

bigram/trigramの候補抽出後に全コーパスを再走査するため、収録閾値以上の頻度は
近似値ではなく全量の正確な値です。サンプリングや先頭文書だけの処理は行いません。

展開後の各行はMozcのシステム辞書ソースと同じ5列です。

```text
よみ<TAB>左文脈ID<TAB>右文脈ID<TAB>コスト<TAB>表記
```

2語・3語は読みと表記を連結した句エントリです。左文脈IDは先頭語、右文脈IDは
末尾語から取得します。コストは各n-gram次数内の観測確率から算出します。これは
4列のMozcユーザー辞書エクスポートではなく、Mozc本体をビルドするときに使う
システム辞書ソースです。必ず同梱の`mozc-id.def`と組み合わせてください。

## Hiragana-to-English dictionary

EDRDGが日次配布する英語版JMdictから、カタカナだけで構成される読みと英語訳を
抽出します。読みはNFKC相当の互換分解・再合成後にひらがなへ変換します。
直接辞書では、CMUdictの発音が読み全体と一致する候補だけを残し、複合語とCMUdict
にない略語は完全一致 allowlist で明示したものだけを残します。JMdictの読み制限、
語彙優先度、語義順はMozcコストへ反映します。

```text
あーと<TAB>left_id<TAB>right_id<TAB>cost<TAB>art
こんぴゅーた<TAB>left_id<TAB>right_id<TAB>cost<TAB>computer
```

同じ読みの自然な候補は残しますが、`ぎゃらりー`に対する`art gallery`や
`corridor`のように読み全体に対応しない候補は収録しません。
入力に使ったJMdict原本、SHA-256、ETag、生成日、ライセンスをReleaseへ同梱します。

## Development

Python 3.11以上、Git、`zstd` が必要です。ランタイムのPython外部依存は
ありません。

```console
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m japanese_corpus discover --output work/discovery.json
cargo test --locked --manifest-path ngram-builder/Cargo.toml
cargo test --locked --manifest-path conversion-engine/Cargo.toml
cargo test --locked --manifest-path jmdict-builder/Cargo.toml
```

## Kana-kanji conversion

生成したunigram／bigram／trigram辞書を直接読み込む、独立したかな漢字変換エンジンを
`conversion-engine`に収録しています。入力読みのラティスを作り、辞書コストに基づく
ビーム探索で文脈を含む句候補を選択します。Mozc本体のビルドは不要です。

```console
cargo build --release --locked --manifest-path conversion-engine/Cargo.toml
conversion-engine/target/release/kana-kanji-converter \
  --dictionary path/to/extracted-dictionary \
  -n 10 こんぴゅーた
```

出力ディレクトリに`mozc-english-reading-unigram-*`があれば、`computer`などの英語候補も
日本語候補と一緒に返ります。`-n 5`で複数候補、`--details`で採用した
1/2/3-gramの分割とコストを確認できます。
引数の読みを省略すると標準入力を1行ずつ変換するため、辞書を一度だけロードして
継続利用できます。詳しくは[`conversion-engine/README.md`](conversion-engine/README.md)
を参照してください。

## Conversion accuracy

[AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench)の
`JWTD_v2/v1/evaluation_items.json`全200件を、公式定義と同じ完全一致の
Accuracy@1および許容正解候補に対する最小文字誤り率（MinCER）で評価します。
データはcommit `401666cd56d1a570c2021798b64b6da4396bfd45`とSHA-256を固定し、
各Releaseに集計レポートを含めます。評価データ本体は再配布しません。

現在の変換APIは左文脈を受け取らないため、AJIMEE-Benchの`context_text`は変換に
使用せず、レポートの`context_mode`を`ignored`としています。その状態を隠さず、
全体に加えて文脈あり100件／なし100件の結果を別々に出力します。Release生成時は
Accuracy@1 0.50以上、平均MinCER 0.09以下を必須とし、精度回帰を検出します。

既存の辞書を手元で評価する例:

```console
cargo run --release --locked \
  --manifest-path conversion-engine/Cargo.toml --bin ajimee-bench -- \
  --dictionary path/to/extracted-release \
  --dataset path/to/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json \
  --output ajimee-bench-report.json \
  --benchmark-commit 401666cd56d1a570c2021798b64b6da4396bfd45
```

個別のWikipediaシャード生成:

```console
PYTHONPATH=src python3 -m japanese_corpus build-wikipedia \
  work/input.json.bz2 work/wikipedia.jsonl.zst \
  --dump-date 20260719 --stats work/wikipedia.stats.json
```

## Automation

GitHub Actionsは毎週月曜03:17 UTCに最新ソースを確認します。Wikipediaの各
シャードを並列処理した後、全コーパスから1/2/3-gram辞書を生成し、完成した
辞書をAJIMEE-Benchで評価してから、完成したdraft Releaseを検証して公開します。
公開後に以前のReleaseとタグを削除するため、未完成ビルドが現在の配布を壊す
ことはありません。同一ソース・同一生成コードなら定期実行はスキップします。

## License

- パイプラインコード: [MIT License](LICENSE)
- 生成コーパス: [CC BY-SA 4.0](LICENSE-DATA.md)
- 生成日英辞書と同梱JMdict: EDRDGのCC BY-SA 4.0
- 出典と権利情報: [NOTICE.md](NOTICE.md)
