kaggle 上基本的房价预测分析，第一次完完整整的将代码实现了一遍，
纪念一下2018.12.16，come on,just do it! The step as follows:
1.建立工程，导入sklearn、pandas等所需要的库
2.读取train.csv,设置目标函数为y=home_data.SalePrice,选取特征变量features给X
3.将数据随机分为验证集与训练集random_state=1
4.建立随机决策树model,random_state=1,训练数据，验证预测，计算平均绝对误差
5.建立指定最大叶子节点max_leaf_nodes=100的决策树模型，训练、验证、误差计算
6.建立随机森林模型，random_state=1,训练，验证、误差计算
7.建立训练全部数据的RandomForestRegressor,random_state=0,同上
8.选择误差最小的模型，对test.csv训练，得出结果，提交结果。

模型效果比较：
Validation MAE when not specifying max_leaf_nodes:29,653
Validation MAE for best value of max_leaf_nodes:27,283
Validation MAE for Random Forest Model: 22,762
Validation MAE on full data for Random Forest Model: 23,040.02603
