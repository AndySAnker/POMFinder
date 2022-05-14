from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import argparse, h5py, os, re, pkg_resources

_BANNER = """
This is a package which takes a directory of PDF files
or a specific file. It then determines the best structural
candidates based of a POM cluster catalog. Results can
be compared with precomputed PDF through fitting.
"""

parser = argparse.ArgumentParser(prog='POMFinder',
                        description=_BANNER, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

requiredNamed = parser.add_argument_group('required named arguments')

requiredNamed.add_argument("-d", "--data", default=None, type=str,
                    help="a directory of PDFs or a file.", required=True)

requiredNamed.add_argument("-n", "--nyquist", default="No", type=str,
                    help="is the data nyquist sampled", required=True)

parser.add_argument("-i", "--Qmin", default=0.7, type=float,
                    help="Qmin value of the experimental PDF")

parser.add_argument("-a", "--Qmax", default=30, type=float,
                    help="Qmax value of the experimental PDF")

parser.add_argument("-f", "--file_name", default='', type=str,
                    help="Name of the output file")

parser.add_argument("-m", "--Qdamp", default=0.04, type=float,
                    help="Qdamp value of the experimental PDF")

def main(args=None):
    args = parser.parse_args(args=args)
    y, y_onehotenc_cat, y_onehotenc_values, POMFinder = get_POMFinder()
    r, Gr = PDF_Preparation(args.data, args.Qmin, args.Qmax, args.Qdamp, rmax=10, nyquist=args.nyquist)
    res, y_pred_proba = POMPredicter(POMFinder, Gr, y_onehotenc_cat);



def get_POMFinder():
    # Get file paths
    load_files = pkg_resources.resource_listdir(__name__, 'Backend/')
    DataBase_path = pkg_resources.resource_filename(__name__, "Backend/"+load_files[0])
    POMFinder_path = pkg_resources.resource_filename(__name__, "Backend/"+load_files[1])
    # Import the Database    
    hf_name = h5py.File(DataBase_path, "r")
    y = hf_name.get('y')
    enc = OrdinalEncoder()
    y_onehotenc_cat = enc.fit(np.array(y))
    y_onehotenc_values = enc.fit_transform(np.array(y))

    # Import POMFinder
    POMFinder = xgb.XGBClassifier()
    POMFinder.load_model(POMFinder_path)
    return y, y_onehotenc_cat, y_onehotenc_values, POMFinder


def PDF_Preparation(Your_PDF_Name, Qmin, Qmax, Qdamp, rmax, nyquist, plot=True):
    for i in range(1000):
        with open(Your_PDF_Name, "r") as file:
            data = file.read().splitlines(True)
            if len(data[0]) == 0:
                with open(Your_PDF_Name, 'w') as fout:
                    fout.writelines(data[1:])
                break
            first_line = data[0]
            if len(first_line) > 3 and re.match(r'^-?\d+(?:\.\d+)?$', first_line[0]) != None and re.match(r'^-?\d+(?:\.\d+)?$', first_line[1]) == None and re.match(r'^-?\d+(?:\.\d+)?$', first_line[2]) != None:
                PDF = np.loadtxt(Your_PDF_Name)
                break
            else:
                with open(Your_PDF_Name, 'w') as fout:
                    fout.writelines(data[1:])
        
    r, Gr = PDF[:,0], PDF[:,1]
    if r[0] != 0: # In the case that the Data not start at 0.
      Gr = Gr[np.where(r==1)[0][0]:] # Remove Data from 0 to 0.5 AA
      Gr = Gr[::10] # Nyquist sample the rest of the Data
      Gr = np.concatenate(([0,0,0,0,0,0,0,0,0,0], Gr), axis=0) # Concatenate 0 - 0.5 AA on the Gr.
    if nyquist == "No" or nyquist == "no":
      Gr = Gr[::10] # Nyquist sample Data
    if len(Gr) >= (rmax*10+1):
      Gr = Gr[:(rmax*10+1)] # In the case Data is up to more than 30 AA, we do not use it.
    else:
      Gr = np.concatenate((Gr, np.zeros((101-len(Gr),))), axis=0) # In case Data is not going to 30 AA, we add 0's.

    Gr[:10] = np.zeros((10,))
    r = np.arange(0, (rmax+0.1), 0.1)
    # Normalise it to the data from the database
    Gr /= np.max(Gr)
    # Add experimental parameters to the Gr
    Gr = np.expand_dims(np.concatenate((np.expand_dims(Qmin, axis=0), np.expand_dims(Qmax, axis=0), np.expand_dims(Qdamp, axis=0), Gr), axis=0), axis=0)

    if plot:
        # Plot the transformation to make sure everything is alright
        plt.plot(PDF[:,0], PDF[:,1], label="Original Data")
        plt.plot(r, Gr[0,3:], label="Gr ready for ML")
        plt.legend()
        plt.title("Original Data vs. normalised Data")
        plt.xlabel("r (AA)")
        plt.ylabel("Gr")
        plt.show()
    
    return r, Gr

def POMPredicter(POMFinder, Gr, y_onehotenc_cat):
    y_pred_proba = POMFinder.predict_proba(Gr);
    y_pred_proba = y_pred_proba[:,1];
    #print (np.shape(y_pred_proba))
    #y_pred_proba = y_pred_proba[0];
    res = sorted(range(len(y_pred_proba)), key = lambda sub: y_pred_proba[sub]);
    res.reverse();
    print ("The 1st guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[0]])[2:-2]+"cale.xyz")
    print ("The 2nd guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[1]])[2:-2]+"cale.xyz")
    print ("The 3rd guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[2]])[2:-2]+"cale.xyz")
    print ("The 4th guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[3]])[2:-2]+"cale.xyz")
    print ("The 5th guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[4]])[2:-2]+"cale.xyz")
    
    return res, y_pred_proba


