# AGENTS.md

鏈枃浠跺畾涔?`E:\\tsfm_projects\\tsproj_stat` 鐨勯」鐩崗浣滆鑼冦€?
## 1. 涓荤嚎杈圭晫

- 涓荤嚎鍛藉悕绌洪棿锛歚app / config / models / evaluation / data_provider / features / eda`
- 缁熶竴鍏ュ彛锛歚run.py`锛堝畬鏁?CLI锛変笌 `main.py`锛堟渶灏忕ず渚嬶級
- `todo_models_source/`銆乣todo_ts_eda/` 宸插畬鎴愯縼绉诲苟鍒犻櫎

## 2. 寮€鍙戠害瀹?
- 缁熻绛栫暐涓荤嚎锛歚one_step` / `recursive` / `direct` + `rolling_backtest`
- 鏂版ā鍨嬪繀椤绘帴鍏?`models/factory.py` 骞跺疄鐜扮粺涓€ `fit/predict`
- 鏂拌瘖鏂兘鍔涘繀椤绘帴鍏?`eda/pipeline.py` 骞惰緭鍑虹粨鏋勫寲缁撴灉
- 瓒嬪娍鍘婚櫎銆侀€嗗彉鎹€佸幓鍣粺涓€鏀惧湪 `data_provider/data_processor.py`

## 3. 渚濊禆涓庤川閲?
- 渚濊禆绛栫暐锛氱‖渚濊禆锛堢己澶卞嵆澶辫触锛?- 渚濊禆鏉ユ簮锛歚pyproject.toml`
- 楠岃瘉鍩虹嚎锛歚pytest -q`

## 4. 鏂囨。鍚屾

姣忔鏀瑰姩鍚庡悓姝ユ鏌ュ苟鏇存柊锛?- `README.md`锛堝姛鑳姐€佺敤娉曘€佷骇鐗┿€侀獙璇佸懡浠わ級
- `AGENTS.md`锛堣竟鐣屻€佹祦绋嬨€佽川閲忛棬妲涳級

## 5. 瀹夊叏涓庨闄?
- 鏈粡纭涓嶆墽琛岀牬鍧忔€ф搷浣?- 涓嶇‖缂栫爜瀵嗛挜鎴栧嚟璇?- 鎺ュ彛鍙樻洿闇€琛ュ厖鏈€灏忓繀瑕佹祴璇?
## 6. 褰撳墠鐘舵€侊紙2026-03-15锛?
- 涓荤嚎鍏ュ彛宸茬粺涓€锛屾敮鎸?`--do-eda`
- EDA 瀛愮郴缁熷凡骞跺叆涓绘祦绋嬶紝杈撳嚭缁撴瀯鍖栨姤鍛婁笌鍥捐〃
- 鏁版嵁棰勫鐞嗗凡鏂板 `DataProcessor`锛堝幓鍣?鍘昏秼鍔?閫嗗彉鎹級
- `BayesianTMT` / `RAR` 宸插畬鎴愰潪鍗犱綅瀹炵幇骞剁撼鍏ユ祴璇曞熀绾?- `statsmodels` 甯歌鎷熷悎鍣０璀﹀憡宸插湪鍏ュ彛瀹氬悜杩囨护