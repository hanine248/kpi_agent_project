def apply_rules(data):
    explanations = []
    adjustment = 0.0

    if data['Overtime_Hours'] > 20:
        explanations.append("High overtime may affect performance negatively.")
        adjustment -= 0.4

    if data['Training_Hours'] < 10:
        explanations.append("Low training hours; development needed.")
        adjustment -= 0.3

    if data['Promotions'] > 2:
        explanations.append("High promotion rate indicates strong performance.")
        adjustment += 0.3

    if data['Monthly_Salary'] > 15000:
        explanations.append("High salary; expectations are higher.")
    
    return explanations, adjustment
