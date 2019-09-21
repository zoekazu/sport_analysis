# Sport Analisys

Collection of sport analysis modules.

## Installing (自環境へのインストール方法)

- `sport_analysis`以下に実装されているライブラリを使用したい場合

```bash
$ pip install git+https://github.com/zoekazu/sport_analysis.git
```

ブランチの指定

```bash
$ pip install git+https://github.com/zoekazu/sport_analysis.git@develop
```

## Contributing（本リポジトリの開発方法）

### First time setup

- Download and install the latest version of git.

- Configure git with your username and email.

```bash
$ git config --global user.name 'your name'
$ git config --global user.email 'your email'
```

- Clone the main repository locally.
  - developブランチをclone
```bash
$ git clone -b develop https://github.com/zoekazu/sport_analysis.git
$ cd sport_analysis
```

- Install the development dependencies

```bash
$ cd sport_analysis
$ pip install -e .[develop]
```

- Install the pre-commit hooks.
```bash
$ pre-commit install
```

### Start codings

- fetch repository
```bash
$ git fetch --all
```

- create your branch
```bash
$ git checkout -b <your-branch-name> origin/develop
```

### Codings

- 文字コードの注意
  - pythonファイル等に日本語を書き込まないでください
    - WinとLinuxの双方の動作保証ができなくなる
    - 誰かが，誤った文字コードの取り扱いをして，バグるのを防ぐ
  - 本リポジトリにて，日本語の書き込みを禁止するファイル
    - `sport_analysis/`以下のpythonファイル
    - `.pre-commit-config.yaml`
    - `setup.cfg`
    - `setup.py`
    - `.gitignore`
  - その他ファイルは日本語OK
    - 例
      - `CHANGESLOG.md`
      - `README.md`
- Gitのコミットの仕方
  - git-flowに従って，ブランチを切ると，わかりやすくなると思います
    - https://danielkummer.github.io/git-flow-cheatsheet/
    - git-flowフローの説明
      - 新規機能開発をする場合は`feature/<hogehoge>`というブランチを切り，その後，`develop`へマージしていきます
  - GUIベース
    - 基本的には，vscodeの左側のタブから，GUIベースで操作するとわかりやすいと思う
    - 「vscode git」とかで検索すれば，

### Running the tests

Run the basic test suite with pytest.

```bash
$ pytest --doctest-modules
```
