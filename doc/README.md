# Documentation


```sh
# ktch/doc
poetry run make html
```



## dev notes

Git repoをリセットするタイミングで以下は削除 or Wiki，docなどへ移行

### CI tools

* Circle CI
* CodeCov
* GitHub Actions

は順次導入．
それ以外はpublicにするタイミングで導入する．

- [ ] ~~Travis CI: free~~
	* is used to test the package in Linux. You need to activate Travis CI for your own repository. Refer to the Travis CI documentation.
- [ ] ~~AppVeyor: public repo, free~~ -> Change to GitHub Actions
	* is used to test the package in Windows. You need to activate AppVeyor for your own repository. Refer to the AppVeyor documentation.
- [x] GitHub Actions due by 2022.8.19
	* is used to test the package in Windows, macOS, and Linux.
- [x] Circle CI: free, due by 2020.12.05 -> to build and deply the documentation.
	* is used to check if the documentation is generated properly. You need to activate Circle CI for your own repository. Refer to the Circle CI documentation.
- [ ] ~~ReadTheDocs~~
	* is used to build and host the documentation. You need to activate ReadTheDocs for your own repository. Refer to the ReadTheDocs documentation.
- [ ] CodeCov: free, due by 2022.9.2
	* for tracking the code coverage of the package. You need to activate CodeCov for you own repository.
- [ ] PEP8Speaks: public repo, free
	* for automatically checking the PEP8 compliance of your project for each Pull Request.

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