# Getting Started

## Clone repo
- Clone repo with the following command: `git clone https://github.com/cjm715/game-of-life.git`

## Downloading competition data
- First agree to conditions on competition site; https://www.kaggle.com/c/conways-reverse-game-of-life-2020
- Follow instructions here to setup permissions and acquiring `kaggle` api command tool: https://www.kaggle.com/docs/api
- Within root of repo, run command: `kaggle competitions download -c conways-reverse-game-of-life-2020`
- Within root of repo, create directory `data` with command : `mkdir data`
- Unzip into `data` folder with command: `unzip conways-reverse-game-of-life-2020 -d data`