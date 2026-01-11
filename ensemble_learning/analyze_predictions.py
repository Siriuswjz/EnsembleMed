"""
分析多个模型的预测差异
判断是否有互补性，是否值得加入集成
"""

import pandas as pd
import numpy as np
from collections import Counter


def load_predictions():
    """加载所有模型的预测结果"""
    models = {
        'best_0.89': './best_0.89/0.89.csv',
        'best_0.90': './best_0.90/submission_pro.csv',
        'metric_v1_0.92': './ensemble_learning/0.92.csv',
        'metric_v3_0.82': './ensemble_learning/0.82.csv',
    }
    
    predictions = {}
    for name, path in models.items():
        try:
            df = pd.read_csv(path)
            predictions[name] = df['label'].values
            print(f"✓ 加载 {name}: {len(df)} 样本")
        except Exception as e:
            print(f"✗ 无法加载 {name}: {e}")
    
    return predictions


def analyze_agreement(predictions):
    """分析模型之间的一致性"""
    print("\n" + "="*60)
    print("模型预测一致性分析")
    print("="*60)
    
    model_names = list(predictions.keys())
    n_samples = len(predictions[model_names[0]])
    
    # 两两比较
    print("\n两两一致率:")
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            agree = np.sum(predictions[name1] == predictions[name2])
            agree_rate = agree / n_samples * 100
            print(f"  {name1} vs {name2}: {agree}/{n_samples} ({agree_rate:.1f}%)")
    
    # 三模型一致性
    if len(model_names) == 3:
        all_agree = np.sum((predictions[model_names[0]] == predictions[model_names[1]]) & 
                          (predictions[model_names[1]] == predictions[model_names[2]]))
        print(f"\n三模型完全一致: {all_agree}/{n_samples} ({all_agree/n_samples*100:.1f}%)")


def analyze_complementarity(predictions):
    """分析互补性"""
    print("\n" + "="*60)
    print("互补性分析")
    print("="*60)
    
    model_names = list(predictions.keys())
    n_samples = len(predictions[model_names[0]])
    
    # 找最好的两个模型
    if len(model_names) >= 2:
        # 假设按准确率排序，取最好的两个
        best_model = 'metric_v1_0.92'
        second_best = 'best_0.90'
        
        if best_model in predictions and second_best in predictions:
            pred_best = predictions[best_model]
            pred_second = predictions[second_best]
            
            # 找出两个最好模型预测不一致的样本
            disagree = pred_best != pred_second
            n_disagree = np.sum(disagree)
            
            print(f"\n{best_model} 和 {second_best} 预测不一致的样本: {n_disagree}")
            
            if n_disagree > 0:
                # 看其他模型在这些争议样本上的表现
                for other_model in model_names:
                    if other_model not in [best_model, second_best]:
                        pred_other = predictions[other_model][disagree]
                        pred_best_disagree = pred_best[disagree]
                        pred_second_disagree = pred_second[disagree]
                        
                        agree_with_best = np.sum(pred_other == pred_best_disagree)
                        agree_with_second = np.sum(pred_other == pred_second_disagree)
                        unique = np.sum((pred_other != pred_best_disagree) & 
                                      (pred_other != pred_second_disagree))
                        
                        print(f"\n{other_model} 在争议样本中:")
                        print(f"  与 {best_model} 一致: {agree_with_best}/{n_disagree} ({agree_with_best/n_disagree*100:.1f}%)")
                        print(f"  与 {second_best} 一致: {agree_with_second}/{n_disagree} ({agree_with_second/n_disagree*100:.1f}%)")
                        print(f"  有独特预测: {unique}/{n_disagree} ({unique/n_disagree*100:.1f}%)")


def simulate_voting(predictions):
    """模拟不同投票策略的结果"""
    print("\n" + "="*60)
    print("投票策略模拟")
    print("="*60)
    
    model_names = list(predictions.keys())
    n_samples = len(predictions[model_names[0]])
    
    # 策略1: 只用前两个模型
    if len(model_names) >= 2:
        pred1 = predictions[model_names[0]]
        pred2 = predictions[model_names[1]]
        
        # 硬投票
        vote_2models = ((pred1 + pred2) >= 1).astype(int)
        
        print(f"\n策略1: 只用 {model_names[0]} + {model_names[1]}")
        print(f"  预测分布: Normal={np.sum(vote_2models==0)}, Disease={np.sum(vote_2models==1)}")
    
    # 策略2: 用全部三个模型
    if len(model_names) == 3:
        pred1 = predictions[model_names[0]]
        pred2 = predictions[model_names[1]]
        pred3 = predictions[model_names[2]]
        
        # 硬投票 (多数投票)
        vote_3models = ((pred1 + pred2 + pred3) >= 2).astype(int)
        
        print(f"\n策略2: 用全部三个模型 (多数投票)")
        print(f"  预测分布: Normal={np.sum(vote_3models==0)}, Disease={np.sum(vote_3models==1)}")
        
        # 比较差异
        diff = np.sum(vote_2models != vote_3models)
        print(f"\n两种策略预测差异: {diff}/{n_samples} ({diff/n_samples*100:.1f}%)")
        
        # 看看哪些样本的预测改变了
        changed_indices = np.where(vote_2models != vote_3models)[0]
        if len(changed_indices) > 0:
            print(f"\n预测改变的样本:")
            for idx in changed_indices[:10]:  # 只显示前10个
                print(f"  样本 {idx}: 2模型={vote_2models[idx]}, 3模型={vote_3models[idx]} "
                      f"(模型1={pred1[idx]}, 模型2={pred2[idx]}, 模型3={pred3[idx]})")
            if len(changed_indices) > 10:
                print(f"  ... 还有 {len(changed_indices)-10} 个样本")
    
    # 策略3: 加权投票 (按准确率加权)
    if len(model_names) == 3:
        weights = [0.90, 0.92, 0.82]  # 对应的准确率
        weighted_vote = (pred1 * weights[0] + pred2 * weights[1] + pred3 * weights[2]) / sum(weights)
        vote_weighted = (weighted_vote >= 0.5).astype(int)
        
        print(f"\n策略3: 加权投票 (权重: {weights})")
        print(f"  预测分布: Normal={np.sum(vote_weighted==0)}, Disease={np.sum(vote_weighted==1)}")
        
        diff_weighted = np.sum(vote_2models != vote_weighted)
        print(f"  与策略1差异: {diff_weighted}/{n_samples} ({diff_weighted/n_samples*100:.1f}%)")


def recommendation(predictions):
    """给出建议"""
    print("\n" + "="*60)
    print("建议")
    print("="*60)
    
    model_names = list(predictions.keys())
    
    if len(model_names) < 3:
        print("模型数量不足，无法分析")
        return
    
    pred1 = predictions[model_names[0]]
    pred2 = predictions[model_names[1]]
    pred3 = predictions[model_names[2]]
    
    # 计算一致性
    agree_12 = np.sum(pred1 == pred2) / len(pred1)
    agree_13 = np.sum(pred1 == pred3) / len(pred1)
    agree_23 = np.sum(pred2 == pred3) / len(pred2)
    
    # 计算第三个模型的差异性
    diversity = 1 - max(agree_13, agree_23)
    
    print(f"\n第三个模型 ({model_names[2]}) 的差异性: {diversity*100:.1f}%")
    
    if diversity < 0.05:
        print("\n❌ 不建议加入第三个模型")
        print("   原因: 与高准确率模型预测太相似，没有带来新信息")
    elif diversity > 0.20:
        print("\n❌ 不建议加入第三个模型")
        print("   原因: 准确率太低(0.82 vs 0.92)，差异太大可能是噪声")
    else:
        print("\n⚠️  可以尝试加入第三个模型")
        print("   原因: 有一定差异性，可能在部分样本上有互补作用")
        print("   建议: 使用加权投票，给第三个模型较小的权重")
    
    print(f"\n推荐策略:")
    print(f"  1. 保守: 只用前两个高准确率模型 (0.90 + 0.92)")
    print(f"  2. 激进: 三模型加权投票，权重 [0.90, 0.92, 0.82]")


def main():
    print("="*60)
    print("模型预测差异分析")
    print("="*60)
    
    # 加载预测
    predictions = load_predictions()
    
    if len(predictions) < 2:
        print("\n错误: 至少需要2个模型的预测结果")
        return
    
    # 分析
    analyze_agreement(predictions)
    analyze_complementarity(predictions)
    simulate_voting(predictions)
    recommendation(predictions)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
