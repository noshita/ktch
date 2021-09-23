# Documentation


```sh
# ktch/doc
poetry run make html
```

## API Reference

Run the following command when a new module is added.

```sh
# ktch
poetry run sphinx-apidoc -o doc/modules/generated  ktch
```