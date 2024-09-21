import numpy as np 
import pandas as pd
import statsmodels.formula.api as smf

#Comment

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
    df = pd.DataFrame({'Sector': ['It', 'It', 'Finance', 'Finance', 'Construction', 'Construction'], 'Headquarters': ['Rotterdam', 'Rotterdam', 'Amsterdam', 'Hague', 'Hague', 'Amsterdam'], "Factor1": [0.6, 0.3, 0.2, 0.6, 0.75, 0.9], "Y":[0.15, 0.3, 0.45, 0.6, 0.75, 0.9]})
    
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


df = add_dummies()
dfX = df.loc[:, ~df.columns.isin(['Y'])]
dfx = df[["Sector_Finance", "Sector_It", "Headquarters_Hague", "Headquarters_Rotterdam", "Factor1"]]

EstOLS(df['Y'], dfX)

model = smf.ols(formula='Y ~ Sector_Finance + Sector_It  + Headquarters_Hague + Headquarters_Rotterdam + Factor1', data=df).fit()

# Print the summary of the model
print(model.summary())
