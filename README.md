[![](https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/project-template) [![](https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true)](https://ci.appveyor.com/project/glemaitre/project-template) [![](https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg)](https://codecov.io/gh/scikit-learn-contrib/project-template) [![](https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master) [![](https://readthedocs.org/projects/sklearn-template/badge/?version=latest)](https://katatch.readthedocs.io/en/latest/?badge=latest)

# ktch - A python package for model-based morphometrics

**ktch** is a python package for model-based morphometrics.


## dev notes

Git repoをリセットするタイミングで以下は削除 or Wiki，docなどへ移行

### CI tools

* Travis CI
* Circle CI
* CodeCov

は順次導入．
それ以外はpublicにするタイミングで導入する．

- [ ] Travis CI: free, due by 2021.03.01
	* is used to test the package in Linux. You need to activate Travis CI for your own repository. Refer to the Travis CI documentation.
- [ ] AppVeyor: public repo, free
	* is used to test the package in Windows. You need to activate AppVeyor for your own repository. Refer to the AppVeyor documentation.
- [ ] Circle CI: free, due by 2020.12.05
	* is used to check if the documentation is generated properly. You need to activate Circle CI for your own repository. Refer to the Circle CI documentation.
- [ ] ReadTheDocs 
	* is used to build and host the documentation. You need to activate ReadTheDocs for your own repository. Refer to the ReadTheDocs documentation.
- [ ] CodeCov: free, due by 2020.12.31
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
