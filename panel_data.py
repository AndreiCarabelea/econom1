import numpy as np 
import pandas as pd
import statsmodels.formula.api as smf
import gc



class CONSTANTS:
    FILE_FF = "capmff_2010-2024_ff.csv"
    FILE_PRICES = "capmff_2010-2024_prices.csv"
    FILE_SECTOR = "capmff_2010-2024_sector.csv"
    
#comment
def inverse(A):
    
    n = len(A)
    A_inv = np.zeros((n,n))

    for i in range(n):
        #solves through LU decomposition and substitution !!!
        col_i = np.linalg.solve(A, np.identity(n)[i])
        A_inv[:, i] = col_i
        
    return A_inv

def test_inverse(n):
    a_test = np.random.randint(50, size=(n, n))
    print(a_test@inverse(a_test))



def add_dummies(df = pd.DataFrame()):
    
    #to be commented for tests only
    df = pd.DataFrame({'Sector': ['It', 'It', 'Finance', 'Finance', 'Construction', 'Construction'], 'Headquarters': ['Rotterdam', 'Rotterdam', 'Amsterdam', 'Hague', 'Hague', 'Amsterdam'], "C": [1, 1, 1, 1, 1, 1], "Factor1": [0.6, 0.3, 0.2, 0.6, 0.75, 0.9], "Y":[0.15, 0.3, 0.45, 0.6, 0.75, 0.9]})
    
    ommited_dummies = []
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            for var in df[col].unique():
                ommited_dummies.append(col+"_"+var)
    
    df = pd.get_dummies(df, drop_first = True, dtype = float)
    
    for col in df.columns:
        if col in ommited_dummies:
            ommited_dummies.remove(col)

        
    
    print(f"Ommited dummies {ommited_dummies}")
    
    #first 2 rows only
    print(df.head(2))
    
    return df


def EstOLS(Y, X):
    """
    Purpose:
        Run OLS, extracting beta and standard errors

    Inputs:
        vY      iN vector, data
        mX      iN x iK matrix, regressors

    Return values:
        v_beta      iK vector, parameters
        vS      iK vector, standard errors
        mS2     iK x iK matrix, covariance matrix
        variance_error     double, residual variance
    """
    if (isinstance(Y, pd.Series)):
        Y= Y.values
    if (isinstance(X, pd.DataFrame)):
        X= X.values

    (N, K)= X.shape

    XtXi = inverse(X.T@X)
    v_beta= XtXi @ X.T@Y
    vE= Y - X@v_beta
    variance_error= vE.T@vE / (N-K)
    
    #covariance matrix
    sigma_beta = XtXi * variance_error
    
    se_beta = np.sqrt(np.diag(sigma_beta))

    return (v_beta, se_beta, sigma_beta, variance_error)

#
# df = add_dummies()
# dfX = df.loc[:, ~df.columns.isin(['Y'])]
# dfx = df[["Sector_Finance", "Sector_It", "Headquarters_Hague", "Headquarters_Rotterdam", "Factor1", "C"]]
#
# # EstOLS(df['Y'], dfX)
#
# model = smf.ols(formula='Y ~ Sector_Finance + Sector_It  + Headquarters_Hague + Headquarters_Rotterdam + Factor1 + C', data=df).fit()
#
# # Print the summary of the model
# print(model.summary())


def GroupMeansD(mX, mD):
    """
    Purpose:
      Calculate the group means of mX, where mD indicates groups

    Inputs:
      mX        iN x iK matrix of data
      mD        iN x iM matrix, dummy indicators

    Return value:
      mMu       iN x iK matrix, group means
    """
    mMb= np.linalg.inv(mD.T@mD)@mD.T@mX
    mMu= mD@mMb

    return mMu


class DataFrame_OP:
    def __init__(self):
        self.returns_df = pd.DataFrame()
        self.ff_df = pd.DataFrame()
        self.sector_df = pd.DataFrame()
        self.all_df = pd.DataFrame()
        
        

    def  load_log_returns(self):
        df= pd.read_csv(CONSTANTS.FILE_PRICES, parse_dates= ['Date'], index_col="Date")
        log_returns = np.log(df / df.shift(1))
        #ignore first day, returns are not defined
        df = df.iloc[1:]
        #ignore stocks with no data for prices 
        df = df.dropna(axis=1)
        
        df_melted = df.reset_index().melt(id_vars='Date', var_name='Stock', value_name='Return')

        # Rename 'index' to 'Date'
        df_melted.rename(columns={'index': 'Date'}, inplace=True)

       
        self.returns_df = df_melted
        
    
    def  load_ff(self):
        df= pd.read_csv(CONSTANTS.FILE_FF, parse_dates= ['Date'], index_col="Date")
        #ignore days with no data for ff returns
        df = df.dropna(axis=0)
        
        df["Mkt-RF"] = df["Mkt-RF"] - df["RF"]
        
        df.drop(columns=['RF'], inplace=True)
        
        self.ff_df = df
    
    def load_sectors(self):
        
        sector_dict = {}
        
        #no need for index 
        df= pd.read_csv(CONSTANTS.FILE_SECTOR, sep=',', engine='python')
        #ignore stocks with no sectors defined
        df = df.dropna(axis=0)
        
        df = df.iloc[:, :2]
        
        df.rename(columns={'stock': 'Stock'}, inplace=True)
     
        self.sector_df = df
        
              
    def join_all_data(self): 
        
        assert( len(self.returns_df) > 0 and len(self.ff_df) > 0 and len(self.sector_df) > 0 ) 
        df_merged1 = pd.merge(self.returns_df, self.ff_df, on='Date', suffixes=('_STOCK', '_INTEREST'))
        df_merged2 = pd.merge(df_merged1, self.sector_df, on='Stock')
        
        df_merged2['Stock_sector'] = df_merged2.iloc[:, 1] + '_' + df_merged2.iloc[:, -1]
        df_merged2.drop(columns=[df_merged2.columns[1], df_merged2.columns[-2]], inplace=True)
        
        gc.collect()  
        
        #not needed if use demenead regression    
        D = pd.get_dummies(df_merged2['Stock_sector'], drop_first = True, dtype = float)
        
        #construct DT dummies for time 
        # DT = 
        pass
        
        self.all_df = df_merged2      
        
    
   
do_obj = DataFrame_OP()

do_obj.load_log_returns()
do_obj.load_ff()
do_obj.load_sectors()
do_obj.join_all_data()

