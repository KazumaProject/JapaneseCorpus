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

青空文庫の旧字旧仮名作品も、`metadata.orthography` で判別できる形で保持します。
かな漢字変換モデルを作る段階で用途に合わせて選別してください。

## Development

Python 3.11以上、Git、`zstd` が必要です。ランタイムのPython外部依存は
ありません。

```console
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m japanese_corpus discover --output work/discovery.json
```

個別のWikipediaシャード生成:

```console
PYTHONPATH=src python3 -m japanese_corpus build-wikipedia \
  work/input.json.bz2 work/wikipedia.jsonl.zst \
  --dump-date 20260719 --stats work/wikipedia.stats.json
```

## Automation

GitHub Actionsは毎週月曜03:17 UTCに最新ソースを確認します。Wikipediaの各
シャードを並列処理し、完成したdraft Releaseを検証してから公開します。
公開後に以前のReleaseとタグを削除するため、未完成ビルドが現在の配布を壊す
ことはありません。同一ソース・同一生成コードなら定期実行はスキップします。

## License

- パイプラインコード: [MIT License](LICENSE)
- 生成コーパス: [CC BY-SA 4.0](LICENSE-DATA.md)
- 出典と権利情報: [NOTICE.md](NOTICE.md)
