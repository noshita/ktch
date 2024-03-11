# Documentation


```sh
# ktch/doc
poetry run make html
```
## dev notes
### Conda
`environment.yml`はconda向けの環境設定ファイル．
`poetry2conda`を使って，`pyproject.toml`から生成する方針にする．

```sh
# 出力を確認するとき
poetry run poetry2conda pyproject.toml -

# 生成するとき
poetry run poetry2conda pyproject.toml environment.yml
```

### ライセンスの確認

[liccheck](https://github.com/dhatim/python-license-check)を使っておこなう．

```sh
poetry export > requirements.txt
poetry run liccheck -s pyproject.toml  

```

### Release Please

#### prefix

* fix: which represents bug fixes, and correlates to a SemVer patch.
* feat: which represents a new feature, and correlates to a SemVer minor.
* feat!:, or fix!:, refactor!:, etc., which represent a breaking change (indicated by the !) and will result in a SemVer major.


#### バージョンの変更

gitのcommit bodyに`Release-As: x.x.x`（`x.x.x`は指定したいバージョン）と記載することで，そのバージョンのPRが作成される．

```sh
git commit --allow-empty -m "chore: 🔧 release x.x.x" -m "Release-As: x.x.x"
```

* [How do I change the version number?| Release Please](https://github.com/googleapis/release-please?tab=readme-ov-file#how-do-i-change-the-version-number)