# JapaneseCorpus generated data notice

## Japanese Wikipedia

- Source: <https://dumps.wikimedia.org/other/cirrus_search_index/>
- Project: <https://ja.wikipedia.org/>
- License information: <https://dumps.wikimedia.org/legal.html>
- Applicable license for this distribution: CC BY-SA 4.0
- Copyright: Wikimedia Foundation and Wikipedia contributors

The corpus contains normalized article text, titles, stable page URLs, and
source timestamps. It does not contain images.

## Aozora Bunko

- Source: <https://www.aozora.gr.jp/>
- Official metadata: <https://www.aozora.gr.jp/index_pages/person_all.html>
- File-handling guidelines: <https://www.aozora.gr.jp/guide/kijyunn.html>
- Official source repository: <https://github.com/aozorabunko/aozorabunko>

Only works whose official work and person copyright flags are all `なし` are
included. The exact repository commit and metadata archive checksum are recorded in
each Release manifest.

## Pipeline inspiration

The source selection and future kana-kanji conversion use case were informed by
[Akaza](https://github.com/akaza-im/akaza), an MIT-licensed Japanese input
method. JapaneseCorpus has its own extraction and packaging implementation.

## Pipeline code

The source code and configuration in this repository are licensed under the
MIT License. Generated corpus files are covered by `LICENSE-DATA.md`.
