# backend/rule_engine.py
def evaluate_rule_based_kpi(performance_score):
    if performance_score <= 2:
        return 'Needs Improvement'
    elif 3 <= performance_score <= 4:
        return 'Average'
    else:
        return 'Excellent'

