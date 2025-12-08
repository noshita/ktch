# Documentation

```sh
# ktch/doc
uv run make html
```

## dev notes

### Release

1. Merge PR generated with Release Please
2. build and publish using uv
3. Merge PR of conda-forge/ktch-feedstock

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

## Licese

The documentations are licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) License.

The documentation system may also include some components licensed under open source licenses.

### scikit-learn

The following items forked from [scikit-learn](https://github.com/scikit-learn/scikit-learn) are licensed under the BSD 3-Clause License.

* shinxext/override_pst_pagetoc.py
* _templates/base.rst
