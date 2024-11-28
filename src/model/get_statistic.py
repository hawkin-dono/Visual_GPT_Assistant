import matplotlib.pyplot as plt
import seaborn as sns
import os

def frequent_component_replacement(df, image_folder):
    """
    Calculate and visualize frequently replaced components.

    Args:
    - df (pd.DataFrame): Maintenance data containing 'LK đồng bộ' (Component Name) 
                         and 'LK không thể tháo rời' (Non-removable Component).
    - image_folder (str): Folder path where the generated image will be saved.

    Returns:
    - dict: {
        'LK đồng bộ': dict: Frequency of replaced 'LK đồng bộ' components,
        'LK không thể tháo rời': dict: Frequency of replaced 'LK không thể tháo rời' components.
    }
    """
    component1_frequency = df['LK đồng bộ'].value_counts()
    component2_frequency = df['LK không thể tháo rời'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    axes[0].bar(component1_frequency.head(10).index, component1_frequency.head(10).values, color='skyblue', alpha=0.7)
    axes[0].set_title("Linh Kiện Đồng Bộ", fontsize=16)
    axes[0].set_xlabel("Component", fontsize=14)
    axes[0].set_ylabel("Replacement Count", fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(component2_frequency.head(10).index, component2_frequency.head(10).values, color='orange', alpha=0.7)
    axes[1].set_title("Linh Kiện Không Thể Tháo Rời", fontsize=16)
    axes[1].set_xlabel("Component", fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)

    image_name = "frequency_of_replaced_components.png"

    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, image_name))
    plt.close()

    return {
        'LK đồng bộ': component1_frequency.to_dict(),
        'LK không thể tháo rời': component2_frequency.to_dict(),
    }

def maintenance_effectiveness(df, image_folder):
    """
    Compare failure frequency and downtime after different maintenance types.

    Args:
    - df (pd.DataFrame): Maintenance data containing 'Mã xử lý' (Maintenance Type) and 'Thời gian dừng máy (giờ)' (Downtime).
    - image_folder (str): Folder path where the generated image will be saved.

    Returns:
    - dict: {
        'failure_count': dict: Failure count for each maintenance type,
        'avg_downtime': dict: Average downtime (hours) for each maintenance type.
    }
    """
    effectiveness = df.groupby('Mã xử lý').agg(
        failure_count=('Mã xử lý', 'size'),
        avg_downtime=('Thời gian dừng máy (giờ)', 'mean')
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(effectiveness.index, effectiveness['failure_count'], color='green', label='Failure Count', alpha=0.6)
    ax1.set_xlabel("Maintenance Type", fontsize=14)
    ax1.set_ylabel("Failure Count", fontsize=14, color='green')

    ax2 = ax1.twinx()
    ax2.plot(effectiveness.index, effectiveness['avg_downtime'], color='blue', marker='o', label='Avg Downtime (Hours)')
    ax2.set_ylabel("Average Downtime (Hours)", fontsize=14, color='blue')

    plt.title("Maintenance Effectiveness", fontsize=16)
    fig.tight_layout()

    image_name = "maintenance_effectiveness.png"

    plt.savefig(os.path.join(image_folder, image_name))
    plt.close()

    return {
        'failure_count': effectiveness['failure_count'].to_dict(),
        'avg_downtime': effectiveness['avg_downtime'].to_dict(),
    }

def failure_frequency_by_device_type(df, image_folder):
    """
    Calculate and visualize failure frequency by device type.

    Args:
    - df (pd.DataFrame): Maintenance data containing 'Tên thiết bị'.
    - image_folder (str): Folder path where the generated image will be saved.

    Returns:
    - dict: {
        'frequency': dict: Failure frequency for each device type.
    }
    """
    frequency = df['Tên thiết bị'].value_counts()

    plt.figure(figsize=(10, 6))
    frequency.plot(kind='bar', color='skyblue')
    plt.title("Failure Frequency by Device Type", fontsize=16)
    plt.xlabel("Device Type", fontsize=14)
    plt.ylabel("Failure Count", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    image_name = "failure_frequency_by_device_type.png"

    plt.savefig(os.path.join(image_folder, image_name))
    plt.close()

    return {
        'frequency': frequency.to_dict()
    }

def common_causes_barchart_by_type(df, machine_type, image_folder):
    """
    Create a bar chart of the most common causes for a specific machine type.

    Args:
    - df (pd.DataFrame): Maintenance data containing:
        - 'Tên thiết bị' (Machine Type)
        - 'Mã Nguyên nhân' (Cause Code)
    - machine_type (str): The type of machine to analyze.
    - image_folder (str): Folder path where the generated image will be saved.

    Returns:
    - dict: {
        'cause_frequency': dict: Frequency of causes for the specified machine type.
    }
    """
    machine_data = df[df['Tên thiết bị'] == machine_type]
    description_res = ""
    if machine_data.empty:
        raise ValueError(f"No data found for machine type {machine_type}")

    cause_counts = machine_data['Mã Nguyên nhân'].value_counts()
    for i in cause_counts.index:
        tmp = machine_data.loc[machine_data['Mã Nguyên nhân'] == i]['Nguyên nhân 1']
        description_res += f"mã lỗi {i} tương đương với các nguyên nhân: {tmp.values} xuất hiện {cause_counts[i]} lần" + "\n"

    plt.figure(figsize=(10, 6))
    cause_counts.plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title(f"Common Causes for Machine Type: {machine_type}", fontsize=16)
    plt.xlabel("Cause Code (Mã Nguyên nhân)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()

    image_name = f"common_causes_{machine_type}.png"
    image_folder = os.path.join(image_folder, image_name)

    plt.savefig(image_folder)
    plt.close()
    
    

    return {
        "text": description_res,
        "image": image_folder
    }
