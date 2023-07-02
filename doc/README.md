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