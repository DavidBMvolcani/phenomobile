from .training import training


class anthocyanin_training(training):
    """
    Specialized training class for anthocyanin projects.
    Inherits from base training class and adds anthocyanin-specific filtering logic.
    """
    
    def __init__(self, dataset_name, config=None, fix_method='KEEP ROWS', task='regression', model=None):
        """
        Initialize anthocyanin training class.
        
        PARAMETERS:
        - dataset_name: name of the dataset that will be used for training
        - config: ConfigManager instance for path resolution and categories
        - fix_method: method that will be chosen when dataset has null values 
        - task: string from one of this :['regression' ,'classification'] 
          default value is 'regression'
        - model: you can set the model you want to fit to data from
          options below. if no option will be set - all models be chosen.
        """
        # Call parent constructor
        super().__init__(dataset_name, config, fix_method, task, model)
        
        # Set paths from parent class for easy access
        if config:
            self.home_path = config.config.get('home_path')
            self.datasets_paths = config.config.get('datasets_path')
        
        # Load anthocyanin-specific categories from config
        self.project_name = config.get('metadata.project_name') if config else None
        self.categories = config.get('categories', {}) if config else {}
    
    def filter_df_by_category(self, df, condition, indicator):
        """
        Override with anthocyanin-specific filtering logic.
        
        PARAMETERS:
        - df: dataframe to filter
        - condition: filtering condition (e.g., 'White and Blue Led', 'White Led', 'Shade', 'Control')
        - indicator: column name to filter on
        
        RETURNS:
        - Filtered dataframe
        """
        if condition == 'White and Blue Led':
            values = self.categories.get('RED_white_blue_led_ids', []) + \
                     self.categories.get('GREEN_white_blue_led_ids', [])
        elif condition == 'White Led':
            values = self.categories.get('RED_white_led_ids', []) + \
                     self.categories.get('GREEN_white_led_ids', [])
        elif condition == 'Shade':
            values = self.categories.get('RED_Shade_ids', []) + \
                     self.categories.get('GREEN_Shade_ids', [])
        elif condition == 'Control':
            values = self.categories.get('RED_Control_ids', []) + \
                     self.categories.get('GREEN_Control_ids', [])
        else:
            values = []
        
        # Filter the dataframe
        res_df = df[df[indicator].isin(values)]
        return res_df
