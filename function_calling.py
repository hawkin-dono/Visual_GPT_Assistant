import pandas as pd

import sys
from src.model import train_model, predict, get_statistic



def predict_repair_method(machine_name, machine_id, error_description, maintain_df):
    res = predict.finding_repair_method(
        machine_name= machine_name,
        machine_id= machine_id, 
        error_description= error_description,
        maintain_df= maintain_df,
    )
    return res
def predict_maintenance_and_failure(machine_number, model, threshold, df1, df2, used_categories):
    res = predict.time_to_failure(
        machine_number=machine_number,
        model=model,
        df1=df1,
        df2=df2,
        used_categories=used_categories,
    )
    return res
def covariate_effects_on_machine(machine_number, model, df1, df2, used_categories):
    res = predict.covariate_effects_on_machine(
        machine_number=machine_number,
        model=model,
        df1=df1,
        df2=df2,
        used_categories=used_categories,)
    
def handle_function_calling(func_name, args):
    model, used_categories = train_model.load_model(
        model_path='saved_models/model.pkl',
        categories_path='saved_models/categories.pkl',
    )
    df1 = pd.read_csv('data/machine_info.csv')
    df2 = pd.read_csv('data/maintain.csv')
    
    if func_name == "finding_repair_method":
        machine_name = args['machine_name']
        machine_id = args['machine_id']
        error_description = args['error_description']
        return predict_repair_method(machine_name, machine_id, error_description, df2)
    
    elif func_name == "recommend_maintenance":
        machine_id = args['machine_id']
        return predict_maintenance_and_failure(machine_number= machine_id, model=model, threshold=0.5, df1=df1, df2=df2, used_categories=used_categories)
    # elif func_name == "covariate_effects_on_machine":
    #     machine_id = args['machine_id']
    #     return covariate_effects_on_machine(machine_number= machine_id, model=model, df1=df1, df2=df2, used_categories=used_categories)
    elif func_name == "common_causes_barchart_by_type":
        machine_id = args['machine_name']
        return get_statistic.common_causes_barchart_by_type(df2, machine_name, image_folder="figures")
    else:
        return 'Function name not found'


# res = predict.finding_repair_method(
#     machine_name= "OP4",
#     machine_id= None, 
#     error_description= "roi dao",
#     maintain_df= df2,
# )
# print(res)

# print("-"*20)
# print(predict.time_to_failure(
#     machine_number="VIM 0159",
#     model=model,
#     df1=df1,
#     df2=df2,
#     used_categories=used_categories,
# ))