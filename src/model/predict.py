import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json


from ..utils import sample_process
from ..utils.text_processing import normalizer, jaccard_similarity_2_shingle

def get_description(df, column: list[str] = ["Tên thiết bị", "Số quản lý thiết bị", "Ngày phát sinh", "Nắm bắt hiện tượng", 
                                             "Nguyên nhân 1", "Nguyên nhân 2", "Nguyên nhân gốc", "Xử lý", "Nội dung phòng chống tái phát"]):
    """ 
    return the description of the machine error
    """
    res = ""
    for idx, row in df[column].iterrows():
        res += f"Lỗi thứ {idx + 1}\n"
        for col in column:
            res += f"{col}: {row[col]}\n" if row[col] is not None else ""
        res += "\n"
    return res 

def finding_repair_method(machine_name, machine_id, error_description, maintain_df: pd.DataFrame):
    """ 
    Question answerd: hãy cho tôi biết lỗi này lúc trc được giải quyết bằng cách nào 
    
    Input: 
        machine_name : tên máy, 
        machine_id: mã máy
        error_description: mô tả lõi ví dụ dao cắt bị mòn
    
    Ouptut: thống kê các biện pháp đã được xử lý với máy
    
        Ví dụ: 
        
        Trong quá khứ, máy CNC mã máy: …. với lỗi đã được sửa chữa 5 lần bao gồm các biện pháp 
        
        - biện pháp 1
        - biện pháp 2
        - biện pháp 3
    
    Idea: 
        - truy xuất thông tin lỗi trong dataframe
        """
    machine_data = maintain_df[maintain_df['Tên thiết bị'] == machine_name] if machine_name is not None else maintain_df
    machine_data = maintain_df[maintain_df['Số quản lý thiết bị'] == machine_id ] if machine_id is not None else machine_data
    if machine_data.empty:
        return 'Không tìm thấy thông tin máy'
    
    machine_data['sim_score'] = machine_data['Nắm bắt hiện tượng'].apply(lambda x: jaccard_similarity_2_shingle(x, error_description))
    machine_data = machine_data.loc[machine_data['sim_score'] > 0.5]
    machine_data = machine_data.sort_values(by='sim_score', ascending=False)
    
    description = get_description(machine_data[:5])
    return description
    
    
    
def recommend_maintenance(machine_number, model, threshold, df1, df2, used_categories):
    """
    Recommend the optimal maintenance time for a machine based on a survival threshold using the Cox Proportional Hazards model.

    Args:
    - machine_number (str): The identifier of the machine.
    - model (CoxPHFitter): The trained Cox Proportional Hazards model.
    - threshold (float): Desired survival probability threshold (e.g., 0.8).
    - df1 (pd.DataFrame): Maintenance data.
    - df2 (pd.DataFrame): Machine data.
    - used_categories (dict): Categories used during training.

    Returns:
    - str: A JSON string representing the result.
    {
        "machine_name": "str: The machine's identifier.",
        "threshold": "float: The survival probability threshold.",
        "recommended_time_date": The recommended maintenance date in 'd-m-y' format, or None if beyond threshold.",
        "message": "str: Message indicating if no optimal time was found."
    }

    E.g: 
    {
        'machine_number': 'VIM 0159',
        'threshold': 0.5,
        'recommended_time_date': '17-12-2024'
    }

    """
    try:
        # Fetch and process machine-specific data
        raw_machine_data = sample_process.fetch_raw_machine_data(machine_number, df1, df2)
        machine_data = sample_process.process_raw_machine_data(raw_machine_data, used_categories)
        machine_data = machine_data[model.params_.index]  # Align with model columns


        # Determine baseline date (last maintenance or manufacture date)
        last_maintenance_date = df2.loc[df2['Số quản lý thiết bị'] == machine_number]['Ngày hoàn thành'].max()
        manufacture_date = df1.loc[df1['Số quản lý thiết bị'] == machine_number]['Ngày sản xuất']
        
        baseline_date = (
            last_maintenance_date if pd.notna(last_maintenance_date) else manufacture_date
        )
        baseline_date = datetime.strptime(baseline_date, "%d-%m-%Y")

        # Predict survival function
        survival_function = model.predict_survival_function(machine_data)

        # Find the first time point where survival probability drops below the threshold
        recommended_time = survival_function.loc[survival_function[0] <= threshold].index[0]
        recommended_time_date = (baseline_date + timedelta(days=int(recommended_time))).strftime("%d-%m-%Y")


        if pd.isna(recommended_time):
            return json.dumps({
                "machine_number": machine_number,
                "threshold": threshold,
                "recommended_time_date": None,
                "message": "The machine is expected to survive beyond the threshold."
            }, ensure_ascii=False)

        return json.dumps({
            "machine_number": machine_number,
            "threshold": threshold,
            "recommended_time_date": recommended_time_date
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

def time_to_failure(machine_number, model, df1, df2, used_categories):
    """
    Estimate the expected failure date for a machine  using the Cox Proportional Hazards model.

    Args:
    - machine_number (str): The identifier of the machine.
    - model (CoxPHFitter): The trained Cox Proportional Hazards model.
    - df1 (pd.DataFrame): Maintenance data.
    - df2 (pd.DataFrame): Machine data.
    - used_categories (dict): Categories used during training.

    Returns:
    - str: A JSON string representing the result.
    {
        "machine_number": "str: The machine's identifier.",
        "expected_failure_date": "str: The estimated failure date in 'd-m-y' format."
    }

    E.g: 
    {
        'machine_number': 'VIM 0159', 
        'expected_failure_date': '06-11-2024'
    }

    """
    try:
        # Fetch and process machine-specific data
        raw_machine_data = sample_process.fetch_raw_machine_data(machine_number, df1, df2)
        machine_data = sample_process.process_raw_machine_data(raw_machine_data, used_categories)
        machine_data = machine_data[model.params_.index]  # Align with model columns

        # Determine baseline date (last maintenance or manufacture date)
        last_maintenance_date = df2.loc[df2['Số quản lý thiết bị'] == machine_number, 'Ngày hoàn thành'].max()
        manufacture_date = df1.loc[df1['Số quản lý thiết bị'] == machine_number, 'Ngày sản xuất'].max()
        
        if pd.notna(last_maintenance_date):
            baseline_date = datetime.strptime(last_maintenance_date, "%d-%m-%Y")
        else:
            baseline_date = datetime.strptime(manufacture_date, "%d-%m-%Y")

        # Predict survival function
        survival_function = model.predict_survival_function(machine_data)

        # Convert index to numeric (representing days since baseline)
        time_points = survival_function.index.to_numpy(dtype=float)
        probabilities = survival_function.values.flatten()

        # Calculate expected time to failure using adjusted weighted mean formula
        survival_diff = probabilities[:-1] - probabilities[1:]
        expected_time_to_failure = np.sum(time_points[:-1] * survival_diff)

        # Calculate the expected failure date
        expected_failure_date = (baseline_date + timedelta(days=int(expected_time_to_failure))).strftime("%d-%m-%Y")

        return json.dumps({
            "machine_number": machine_number,
            "expected_failure_date": expected_failure_date
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

def predict_maintenance_and_failure(machine_number, model, threshold, df1, df2, used_categories):
    """
    Predict the optimal maintenance time and expected failure date for a machine using the Cox Proportional Hazards model.

    Args:
    - machine_number (str): The identifier of the machine.
    - model (CoxPHFitter): The trained Cox Proportional Hazards model.
    - threshold (float): Desired survival probability threshold (e.g., 0.8).
    - df1 (pd.DataFrame): Maintenance data.
    - df2 (pd.DataFrame): Machine data.
    - used_categories (dict): Categories used during training.

    Returns:
    - str: A JSON string representing the result.
    {
        "machine_number": "str: The machine's identifier.",
        "threshold": "float: The survival probability threshold.",
        "recommended_time_date": The recommended maintenance date in 'd-m-y' format, or None if beyond threshold.",
        "expected_failure_date": "str: The estimated failure date in 'd-m-y' format."
    }

    E.g: 
    {
        'machine_number': 'VIM 0159',
        'threshold': 0.5,
        'recommended_time_date': '17-12-2024',
        'expected_failure_date': '06-11-2024'
    }"""
    
    predict_maintenance = recommend_maintenance(machine_number, model, threshold, df1, df2, used_categories)
    predict_failure = time_to_failure(machine_number, model, df1, df2, used_categories)
    
    predict_maintenance = json.loads(predict_maintenance)
    predict_failure = json.loads(predict_failure)
    
    return json.dumps({
        "machine_number": machine_number,
        "expected_failure_date": predict_failure['expected_failure_date'],
        "recommended_maintenance_date": predict_maintenance['recommended_time_date'],
    }, ensure_ascii=False)

    
def covariate_effects_on_machine(machine_number, model, df1, df2, used_categories):
    """
    Analyze the impact of each covariate on a machine's risk of failure with enhanced interpretability using the Cox Proportional Hazards model.

    Args:
    - machine_number (str): The identifier of the machine.
    - model (CoxPHFitter): The trained Cox Proportional Hazards model.
    - df1 (pd.DataFrame): Maintenance data.
    - df2 (pd.DataFrame): Machine data.
    - used_categories (dict): Categories used during training.

    Returns:
    - str: A JSON string representing the result.
    {
        "machine_number": "str: The machine's identifier.",
        "covariate_effects": "dict: A dictionary of covariate impacts and description."
    }

    E.g: 
    {
        'machine_number': 'VIM 0159',
        'covariate_effects': {
            'Thời gian dừng máy (giờ)': {
                'value': 2.0,
                'description': "Covariate 'Thời gian dừng máy (giờ)' with value '2.0' decreases failure risk by 14.30%."
            },
            'Số người thực hiện': {
                'value': 1.0,
                'description': "Covariate 'Số người thực hiện' with value '1.0' decreases failure risk by 2.34%."
            },
            'Điện áp tiêu thụ (V)': {
                'value': 220.0,
                'description': "Covariate 'Điện áp tiêu thụ (V)' with value '220.0' decreases failure risk by 0.33%."
            },
            'Tuổi thọ thiết bị': {
                'value': 1249.0,
                'description': "Covariate 'Tuổi thọ thiết bị' with value '1249.0' increases failure risk by 0.21%."
            },
            .......
        }
    }
    """
    try:
        # Fetch and process machine-specific data
        raw_machine_data = sample_process.fetch_raw_machine_data(machine_number, df1, df2)
        machine_data = sample_process.process_raw_machine_data(raw_machine_data, used_categories)
        machine_data = machine_data[model.params_.index]  # Align with model columns

        # Analyze covariate effects
        effects = {}
        for covariate, value in machine_data.iloc[0].items():
            if value != 0:  # Skip zero (inactive) covariates
                effect = model.params_[covariate]
                hazard_ratio = np.exp(effect)
                percent_impact = (hazard_ratio - 1) * 100

                effects[covariate] = {
                    "value": value,
                    "description": f"Covariate '{covariate}' with value '{value}' "
                                   f"{'increases' if effect > 0 else 'decreases'} "
                                   f"failure risk by {abs(percent_impact):.2f}%."
                }

        return json.dumps({
            "machine_number": machine_number,
            "covariate_effects": effects
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})