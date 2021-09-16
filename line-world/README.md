# line-world

環境：一次元の直線格子世界
エージェント：その格子世界を移動するエージェント
状態 s: position(現在地)
行動 a: STEP(position を+1),STOP(position 変化なし)

## Sample

grid=[0,0,1,0,0,1]

- 1: 報酬のありか
- 0: なにもなし

## Files

すべてモデルベース

- env.py: ベースとなる各クラスを定義したもの
- mdp.py: MDP を作成（特にほかでは使われていない）
- planner.py: 価値反復法と方策反復法を実装
- value_dp.py: DP 法で価値を求める実装
