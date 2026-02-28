# Changelog

## [0.7.3](https://github.com/noshita/ktch/compare/v0.7.2...v0.7.3) (2026-02-28)


### Miscellaneous Chores

* 🔧 clean up test directories ([f9897ea](https://github.com/noshita/ktch/commit/f9897ea15c15916590a5121043410e9707cabe89))
* 🔧 remove bottle dataset ([839b595](https://github.com/noshita/ktch/commit/839b595538c67cfa005560008712bdd32246a0ea))
* 🔧 remove update_versions_json.py script ([1ef2f4c](https://github.com/noshita/ktch/commit/1ef2f4cb125a47ac3be8301852caef13f6f3a02e))

## [0.7.2](https://github.com/noshita/ktch/compare/v0.7.1...v0.7.2) (2026-02-20)


### Features

* ✨ update image_passiflora_leaves dataset ([9aae743](https://github.com/noshita/ktch/commit/9aae74366066b24b234202297376cad43d35a342))


### Bug Fixes

* 🐛 fix ReDoS vulnerability in TPS CURVES regex (code-scanning[#14](https://github.com/noshita/ktch/issues/14)) ([a913d89](https://github.com/noshita/ktch/commit/a913d89ff026a9cca5e8d4125dd6d88c4d14d654))


### Documentation

* 📚 dump version ([0aea7ec](https://github.com/noshita/ktch/commit/0aea7ec9b9f4fe0a7a56c42d68e982c3ee852184))
* 📚 update image_passiflora_leaves dataset ([71ce0d3](https://github.com/noshita/ktch/commit/71ce0d3de296d9baa7321131f249ae5a2af1caee))


### Miscellaneous Chores

* 🔧 remove obsolete Poetry-era and conda-related config ([8e72be4](https://github.com/noshita/ktch/commit/8e72be47ea7237819f49d2b0bfe523c939644c52))

## [0.7.1](https://github.com/noshita/ktch/compare/v0.7.0...v0.7.1) (2026-02-19)


### Features

* ✨ lazy import in plot module ([94de8cc](https://github.com/noshita/ktch/commit/94de8cc6c3b2f1c4704047ccc400d34accf9a851))


### Bug Fixes

* 🐛 .release-please-manifest.json ([ab0e443](https://github.com/noshita/ktch/commit/ab0e443e38358ce44c55603070baec7f51b7e9ee))
* 🐛 release-please workflow ([6c751a9](https://github.com/noshita/ktch/commit/6c751a96c3590dd8390b2aec38b1c7d800baf16f))
* 🐛 setup-uv ([31e1b51](https://github.com/noshita/ktch/commit/31e1b51e4e3c0f51e25a6591f53951c6c87f7b38))
* 🐛 sphinx multiversion switcher ([93e1ef3](https://github.com/noshita/ktch/commit/93e1ef369404db9035cb467c199c65698145134b))


### Performance Improvements

* ⚡️ doc build ([2281da7](https://github.com/noshita/ktch/commit/2281da7578d70fcfbe63419e876383897a199826))


### Documentation

* 📚 add guidelines for optional dependencies ([b7628f3](https://github.com/noshita/ktch/commit/b7628f3e62fde9ec394217884cfb250f3d765dda))


### Miscellaneous Chores

* 🔧 change trigger for documentation build ([57f0321](https://github.com/noshita/ktch/commit/57f0321ddde15ea77cd715deb0615f297c6ba692))
* 🔧 scipy &gt;= 1.15 ([0ef8844](https://github.com/noshita/ktch/commit/0ef8844e167059d9f8d4d8f1868c0dc3da888133))
* 🔧 script for doc versions ([641fc97](https://github.com/noshita/ktch/commit/641fc97dd2b02ec9daeba40ef227810dd4273732))
* 🔧 update doc versions ([8b18c47](https://github.com/noshita/ktch/commit/8b18c4797d67630cb000bc54477214794174e1c9))


### Code Refactoring

* ♻️ datasets module ([4b09555](https://github.com/noshita/ktch/commit/4b09555b448feb152f979e4b49b47476294f566f))

## [0.7.0](https://github.com/noshita/ktch/compare/v0.6.1...v0.7.0) (2026-02-12)


### Features

* ✨  add registry update script via R2 manifest.json ([b9d63bc](https://github.com/noshita/ktch/commit/b9d63bcf59121c109097f2e4b950aa3638bd16e1))
* ✨ add new dataset: outline_leaf_bending ([25d9d30](https://github.com/noshita/ktch/commit/25d9d308952ffbc7c0273c9b1b78428fab07ea8f))
* ✨ add normalization with semi-major axis for 3D EFA ([6b3440f](https://github.com/noshita/ktch/commit/6b3440f4a98175fc2b456e70d524d73d01bd92a4))
* ✨ add Passiflora leaf image dataset ([f75cb0c](https://github.com/noshita/ktch/commit/f75cb0c97b5425a84e765000cc04c662f8deb5c1))
* ✨ datasets trilobite cephala ([1ce07aa](https://github.com/noshita/ktch/commit/1ce07aa46134f64c914719c3034417e8be2f7899))
* ✨ n_jobs for GPA ([679dcd2](https://github.com/noshita/ktch/commit/679dcd2954929bbfa5baed55b10ee0bcc17ae3a0))
* ✨ Normalization for 3D EFA ([59ada8f](https://github.com/noshita/ktch/commit/59ada8fde44775e9b4722425b7027afae2d264e7))
* ✨ semilandmark analysis ([a5e6076](https://github.com/noshita/ktch/commit/a5e6076104e5b24529e52a356f8345946d16b973))
* ✨ tutorial for outline extraction ([3bfe708](https://github.com/noshita/ktch/commit/3bfe7084de0f083d2db09c30e8563d950a11a2d1))


### Bug Fixes

* 🐛 add jupytext for v0.6.1 build ([8e7cff2](https://github.com/noshita/ktch/commit/8e7cff25bd3ba5726a2d9949279a55f95fb28b7b))
* 🐛 add jupytext for v0.6.1 build ([932a9ab](https://github.com/noshita/ktch/commit/932a9ab7bc6c29472c900ae2d33660a8b75ad5d8))
* 🐛 add path traversal validation for zip extraction ([01010c4](https://github.com/noshita/ktch/commit/01010c435990252538c2f1ee5a273bf64caaf528))
* 🐛 add version_match logging ([8e7cff2](https://github.com/noshita/ktch/commit/8e7cff25bd3ba5726a2d9949279a55f95fb28b7b))
* 🐛 add version_match logging ([932a9ab](https://github.com/noshita/ktch/commit/932a9ab7bc6c29472c900ae2d33660a8b75ad5d8))
* 🐛 doc ([96ff2c8](https://github.com/noshita/ktch/commit/96ff2c80be52a3d0f58840b684dedb02511eb56d))
* 🐛 remove duplicate data and update metadata and tps data ([c979e80](https://github.com/noshita/ktch/commit/c979e80568de71a360912543929b39166423f22d))
* 🐛 replace exec() with ast-based parsing in update_registry ([abe1973](https://github.com/noshita/ktch/commit/abe1973bcb1b39c03a210634e1ca562e140b8648))
* 🐛 sphinx-multiversion and switcher ([932a9ab](https://github.com/noshita/ktch/commit/932a9ab7bc6c29472c900ae2d33660a8b75ad5d8))
* 🐛 test-codecov.yml ([1bafb2b](https://github.com/noshita/ktch/commit/1bafb2b8b2cc61e8bc92d70a6f5731517aedb237))
* 🐛 tip coordinates in outline_leaf_bending dataset ([f5a7996](https://github.com/noshita/ktch/commit/f5a7996a470bffbc94eaa90e8443794a44f83860))
* 🐛 typo ([1fb2811](https://github.com/noshita/ktch/commit/1fb281108d0a3249adf93db46f033abbfb376d3f))
* 🐛 use sphinx-multiversion on GitHub for env variable ([8e7cff2](https://github.com/noshita/ktch/commit/8e7cff25bd3ba5726a2d9949279a55f95fb28b7b))
* 🐛 use sphinx-multiversion on GitHub for env variable ([932a9ab](https://github.com/noshita/ktch/commit/932a9ab7bc6c29472c900ae2d33660a8b75ad5d8))
* 🐛 version switcher, redirect to stable ([8e7cff2](https://github.com/noshita/ktch/commit/8e7cff25bd3ba5726a2d9949279a55f95fb28b7b))
* 🐛 version switcher, redirect to stable ([932a9ab](https://github.com/noshita/ktch/commit/932a9ab7bc6c29472c900ae2d33660a8b75ad5d8))
* 🐛 version_match ([8e7cff2](https://github.com/noshita/ktch/commit/8e7cff25bd3ba5726a2d9949279a55f95fb28b7b))
* 🐛 version_match ([932a9ab](https://github.com/noshita/ktch/commit/932a9ab7bc6c29472c900ae2d33660a8b75ad5d8))


### Performance Improvements

* ⚡️ enable parallel read (and execute) ([8f1f1ff](https://github.com/noshita/ktch/commit/8f1f1ff176d0c9f30651e60e94fe8d5616de5f8e))


### Documentation

* 📚  move 2D outline registration guide from tutorials to how-to ([cdb1ad3](https://github.com/noshita/ktch/commit/cdb1ad3c4c353d6d5bfd8dc4fa62d16c89088aca))
* 📚 drop doc of v0.6.1 ([5cdb7aa](https://github.com/noshita/ktch/commit/5cdb7aafcae3dfa15af26e66091d8c76f77b6725))
* 📚 redesign documentation with Diátaxis framework and multi-version support ([#108](https://github.com/noshita/ktch/issues/108)) ([7bc8465](https://github.com/noshita/ktch/commit/7bc8465737cc2109a7a61a3607a701435d324ea4))
* 📚 refactor configuration_plot, use vtp files for spharm ([0db89a8](https://github.com/noshita/ktch/commit/0db89a86e0d2633ad6e666543aacb7116003d4d5))
* 📚 reorganize contributing guides and migrate liccheck to licensecheck ([d475742](https://github.com/noshita/ktch/commit/d47574262b451f5e3aeede32a5fd9aabf9116b2e))
* 📚 sitemap ([19a1932](https://github.com/noshita/ktch/commit/19a1932c4a0e482b6e552526dcc709691a249111))
* 📚 Update README.md ([d0d43eb](https://github.com/noshita/ktch/commit/d0d43eb446ad6dd92e1e104d4d82d3fbbce475bd))
* 📚 update tutorial for 3D EFA to cover morphospace reconstruction ([d63fcb3](https://github.com/noshita/ktch/commit/d63fcb3b0395169a92173e64d91c076e2c186602))


### Miscellaneous Chores

* release 0.7.0 ([e9c2476](https://github.com/noshita/ktch/commit/e9c2476ef3790c6afd687052fb39841644c74292))

## [0.6.1](https://github.com/noshita/ktch/compare/v0.6.0...v0.6.1) (2025-12-08)


### Bug Fixes

* 🐛 rename _tps to _kriging ([94089f8](https://github.com/noshita/ktch/commit/94089f83f70a0dc745ff9e8491c083d006d89b9d))


### Documentation

* 📚 update api, notebooks ([63ab4ac](https://github.com/noshita/ktch/commit/63ab4ac3b31ffcdc398406ed0686a520d19943bc))


### Miscellaneous Chores

* release 0.6.1 ([4cb985a](https://github.com/noshita/ktch/commit/4cb985a61221ff510a0af9b867f9823f79e733fe))

## [0.6.0](https://github.com/noshita/ktch/compare/v0.5.0...v0.6.0) (2025-12-04)


### Features

* ✨ add  option to  2D ([6032557](https://github.com/noshita/ktch/commit/6032557d1cce9fc9d06854a54111d6a27353f8ad))
* ✨ Add chain code file I/O functionality ([61905e0](https://github.com/noshita/ktch/commit/61905e00b554785ca1290739971c40ad3bed1dee))
* ✨ Add coordinate conversion functions to ChainCodeData class ([5384d02](https://github.com/noshita/ktch/commit/5384d02aa0170e7ba12f1f4955a963a08076532b))
* ✨ add n_jobs to EllipticFourierAnalysis class ([5db88e5](https://github.com/noshita/ktch/commit/5db88e58cd25f0721962c2991371fdc48bce55ab))
* ✨ add n_jobs to EllipticFourierAnalysis class ([321193a](https://github.com/noshita/ktch/commit/321193aad7770f8958b6261da1be475d8406f497))
* ✨ add plot module ([4f01f3e](https://github.com/noshita/ktch/commit/4f01f3ec439966cc8a8cc61d554093f378154ee0))
* ✨ SphericalHarmonicAnalysis class ([7103a16](https://github.com/noshita/ktch/commit/7103a16f341391f7bc8c389a3ec6473de69ec2ec))
* ✨ Update chain code format to use simplified sample name format ([38a4be4](https://github.com/noshita/ktch/commit/38a4be486ec5007c51baced185d084ef3b6d2cbe))
* ✨ Update chain code implementation to validate direction codes (0-7) ([2242ecc](https://github.com/noshita/ktch/commit/2242ecc6d7938fa5cd97e822bf7b45aee3bdc6e4))
* Add comprehensive input validation and error handling ([e5b7725](https://github.com/noshita/ktch/commit/e5b7725dd75c536ae4156076b81fa7f404e13ed4))
* Add type hints and improve documentation for SPHARM-PDM module ([fc47215](https://github.com/noshita/ktch/commit/fc472150ac0c896a9c2aa8cf72fadd9888ea380b))


### Bug Fixes

* _cvt_spharm_coef_SPHARMPDM_to_list ([5122e8b](https://github.com/noshita/ktch/commit/5122e8b83516cb8198c49ea514a7014cbb360b51))
* 🐛 Add n_samples definition in write_chc function ([7944293](https://github.com/noshita/ktch/commit/7944293f455a05ff4b8dafc623e7c916040c0b5c))
* 🐛 add seaborn for test-codecov ([5972268](https://github.com/noshita/ktch/commit/5972268df9966eeddecafa8a656e31a355639a50))
* 🐛 change outline to harmonic ([65bc786](https://github.com/noshita/ktch/commit/65bc786040631d533873c54802ea6322104f579b))
* 🐛 Fix chain_code property in ChainCodeData class ([7632cbf](https://github.com/noshita/ktch/commit/7632cbf3bf31c6792fbf624dfc84c7fdba5dfca6))
* 🐛 Fix handling of 1D arrays in write_chc function ([a51136e](https://github.com/noshita/ktch/commit/a51136e7fc815f5a3e02b5ed3489ad7656e0c370))
* 🐛 notebook for spharm ([36e17dd](https://github.com/noshita/ktch/commit/36e17ddcb31a561332357f5571b7eda32d6718fb))
* 🐛 notebook for spharm ([c4f26cd](https://github.com/noshita/ktch/commit/c4f26cd02a5eb95d7502ec8f194ad709bab73c06))
* 🐛 Optimize regex patterns and add test for CURVES functionality ([0537c4f](https://github.com/noshita/ktch/commit/0537c4f1c681f52a68e6c9bab4328ab6164402ba))
* 🐛 Optimize regex patterns to avoid catastrophic backtracking ([ee76958](https://github.com/noshita/ktch/commit/ee76958f8eabcd25f4c8e4d0a2734440760eb8ec))
* 🐛 Pass validate parameter to ChainCodeData constructor ([53f100e](https://github.com/noshita/ktch/commit/53f100ef71ba43c577e4f3d114b781ead779431a))
* 🐛 release-please config files ([a231c32](https://github.com/noshita/ktch/commit/a231c32fb686ac6f7ee91682d4147c2085ccbd6f))
* 🐛 remove unused Class ([f506f2a](https://github.com/noshita/ktch/commit/f506f2ac51533f378cd8cda336a38f3ae9916191))
* 🐛 remove unused import ([a2bbb6a](https://github.com/noshita/ktch/commit/a2bbb6a528a95cc5663614d0c8fc88fcdbee101a))
* 🐛 scale X before iterations ([82f4a16](https://github.com/noshita/ktch/commit/82f4a1622b7aeae1d36fca77ef48e0e70bf7360e))
* 🐛 tests for io of SPHARM-PDM ([543c1e9](https://github.com/noshita/ktch/commit/543c1e98464bbd541beb42e7b9965e66fa10251b))


### Performance Improvements

* Optimize coefficient parsing and conversion logic ([3ae8db3](https://github.com/noshita/ktch/commit/3ae8db3281e30defc2ce1a4d0389dbfc33716d64))


### Documentation

* 📚 add api, docstring, and notebook for spharm ([da7a9d9](https://github.com/noshita/ktch/commit/da7a9d910b7a3dcc30cb02de0fa185b5b0d5e064))
* 📚 fix 3D plots using Plotly, add myst-nb config ([c891b03](https://github.com/noshita/ktch/commit/c891b0360343b9ac49b741064a8a7d1cfe51cf76))
* 📚 update api for chc, spharmpdm ([719ef46](https://github.com/noshita/ktch/commit/719ef460fe5500276a50460909db33e776d1a6b0))
* 📚 update notebook for spharm ([a3c5738](https://github.com/noshita/ktch/commit/a3c57386c65d4092e47ba3e66ad5c136d06e1da4))

## [0.5.0](https://github.com/noshita/ktch/compare/v0.4.3...v0.5.0) (2025-02-07)


### Features

* ✨ add example notebook of 3D EFA ([8f8ead2](https://github.com/noshita/ktch/commit/8f8ead20d27fbe86e7c49ff7e1fe8ff1e2f72838))


### Bug Fixes

* 🐛 github actions test-codecov.yml ([30e31b1](https://github.com/noshita/ktch/commit/30e31b1498fc7b4a01e21938fd433e02b0c6ed6b))
* 🐛 paths filter ([3dbf322](https://github.com/noshita/ktch/commit/3dbf3229b5db6d4e425662622ebf4f0b17bfcbe2))


### Documentation

* 📚 reorganize documentations ([ff4ec55](https://github.com/noshita/ktch/commit/ff4ec5546582b11e0f48c396df2a9fc8df2c3b6c))

## [0.4.3](https://github.com/noshita/ktch/compare/v0.4.2...v0.4.3) (2025-02-06)


### Features

* ✨ add 3D EFA ([4d74c92](https://github.com/noshita/ktch/commit/4d74c92949e030703918c26c2c5de7c7f5074d6d))
* ✨ add auto-approve for owener ([a38fb05](https://github.com/noshita/ktch/commit/a38fb05db971cd2e62ba03f938ccd81ac0b19d69))


### Bug Fixes

* 🐛 actions/setup-python ([d06a63e](https://github.com/noshita/ktch/commit/d06a63e8fc558f66d3dcce14393bddbe32b7372e))
* 🐛 github actions filters ([43aa28c](https://github.com/noshita/ktch/commit/43aa28cb75cfeb2791a376b6ad43859167015ac7))
* 🐛 pull_request ([19ad8c7](https://github.com/noshita/ktch/commit/19ad8c757bfd67016ad0ca08fbf165ca79f0264d))


### Miscellaneous Chores

* 🔧  v0.4.3 ([7e07f19](https://github.com/noshita/ktch/commit/7e07f199dfad803844867e6e06def27efbd2a795))
* 🔧 update requirements.txt ([a43ec5f](https://github.com/noshita/ktch/commit/a43ec5f7ec020c338cb04bf87c8f22020c081f60))

## [0.4.2](https://github.com/noshita/ktch/compare/v0.4.1...v0.4.2) (2024-06-09)


### Bug Fixes

* 🐛  inefficient regular expression ([a2d04a2](https://github.com/noshita/ktch/commit/a2d04a2de7bbf204e289e2c2e4cdf75b0a59565f))
* 🐛 actions/setup-python ([d06a63e](https://github.com/noshita/ktch/commit/d06a63e8fc558f66d3dcce14393bddbe32b7372e))
* 🐛 install jupytext ([b938cb1](https://github.com/noshita/ktch/commit/b938cb1e52a77191069538942caeb71f46c07c1f))
* 🐛 invalid escape sequence ([394464d](https://github.com/noshita/ktch/commit/394464d84c1ec0da8fd50c12bd4a93c80afbe31c))


### Documentation

* 📚 introduce jupytext and remove .ipynb files ([d8b38b9](https://github.com/noshita/ktch/commit/d8b38b9a11408da1fd464a0f2c7b268688c23350))
* 📚 introduce jupytext and remove .ipynb files ([0d6dd70](https://github.com/noshita/ktch/commit/0d6dd705a53a52495e15e945612b4185ffd2b0a5))


### Miscellaneous Chores

* 🔧 update requirements.txt ([a43ec5f](https://github.com/noshita/ktch/commit/a43ec5f7ec020c338cb04bf87c8f22020c081f60))
* release 0.4.1 ([cc1c2a6](https://github.com/noshita/ktch/commit/cc1c2a62aeaebf444b14e05285378e6de8463860))

## [0.4.1](https://github.com/noshita/ktch/compare/v0.4.1...v0.4.1) (2024-03-12)


### Bug Fixes

* 🐛  inefficient regular expression ([a2d04a2](https://github.com/noshita/ktch/commit/a2d04a2de7bbf204e289e2c2e4cdf75b0a59565f))
* 🐛 install jupytext ([b938cb1](https://github.com/noshita/ktch/commit/b938cb1e52a77191069538942caeb71f46c07c1f))
* 🐛 invalid escape sequence ([394464d](https://github.com/noshita/ktch/commit/394464d84c1ec0da8fd50c12bd4a93c80afbe31c))


### Documentation

* 📚 introduce jupytext and remove .ipynb files ([d8b38b9](https://github.com/noshita/ktch/commit/d8b38b9a11408da1fd464a0f2c7b268688c23350))
* 📚 introduce jupytext and remove .ipynb files ([0d6dd70](https://github.com/noshita/ktch/commit/0d6dd705a53a52495e15e945612b4185ffd2b0a5))


### Miscellaneous Chores

* release 0.4.1 ([cc1c2a6](https://github.com/noshita/ktch/commit/cc1c2a62aeaebf444b14e05285378e6de8463860))

## [0.3.3](https://github.com/noshita/ktch/compare/v0.3.2...v0.3.3) (2024-03-10)


### Features

* ✨ add to doc ([25317da](https://github.com/noshita/ktch/commit/25317da0805b13316e767a563460c597da2066a9))
* ✨ add tps transformation grid ([ae87df5](https://github.com/noshita/ktch/commit/ae87df55d6f9590ac11f1be94ab493c647f71980))
* ✨ add tps transformation grid ([d295a4f](https://github.com/noshita/ktch/commit/d295a4fbc1629475b1fb4f85a569e4b178704f75))
* ✨ thin-plate spline ([09f8c4a](https://github.com/noshita/ktch/commit/09f8c4a4c7cd6061e8a55939b9cb1801386267ca))

## [0.3.1](https://github.com/noshita/ktch/compare/v0.3.0...v0.3.1) (2024-02-08)


### Bug Fixes

* 🐛 phase shift ([25450bc](https://github.com/noshita/ktch/commit/25450bc4f112f0f81e9ba34d4d832275302320be))
* 🐛 phaseshift ([25450bc](https://github.com/noshita/ktch/commit/25450bc4f112f0f81e9ba34d4d832275302320be))
* 🐛 phaseshift ([25450bc](https://github.com/noshita/ktch/commit/25450bc4f112f0f81e9ba34d4d832275302320be))
* 🐛 setter of SPHARMCoefficients class ([ee12ce2](https://github.com/noshita/ktch/commit/ee12ce2e62243c48ac657b322f62fb9fe0ee4231))


### Documentation

* 📚 add robots.txt ([ff2aeda](https://github.com/noshita/ktch/commit/ff2aedaf031ca032dac0b8daf40c170a40cb1b3a))
* 📚 add sitemap ([8ea64cf](https://github.com/noshita/ktch/commit/8ea64cffe07cac0f3a11734b7c94eaec511e60e1))
* 📚 doc ([ceb1ee3](https://github.com/noshita/ktch/commit/ceb1ee310ce9b00f13b836592ea73a39074b43a9))
* 📚 update autodoc ([ba2c2d6](https://github.com/noshita/ktch/commit/ba2c2d6cc66325fbbf6758a5ad260f1c723fb116))
* 📚 update doc config ([72f6df7](https://github.com/noshita/ktch/commit/72f6df73d8d7ce97644382a6b26e22eefbb00c39))
* 📚 update favicon ([54b6cec](https://github.com/noshita/ktch/commit/54b6cec17a9ac894ebcabfde7514722b4c046712))
* 📚 update favicon ([54b6cec](https://github.com/noshita/ktch/commit/54b6cec17a9ac894ebcabfde7514722b4c046712))
* 📚 update favicon ([54b6cec](https://github.com/noshita/ktch/commit/54b6cec17a9ac894ebcabfde7514722b4c046712))
