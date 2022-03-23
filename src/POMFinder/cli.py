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

requiredNamed.add_argument("-n", "--nyquist", default=True, type=bool,
                    help="Is the data nyquist sampled", required=True)

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
    plot_simulated_predictions(Gr, res, y_onehotenc_cat, args.Qmin, args.Qmax, args.Qdamp);




def get_POMFinder():
    # Import the Database
    cwd = os.getcwd()
    load_files = pkg_resources.resource_listdir(__name__, '../src/')
    print (__name__)
    print (load_files)
    
    hf_name = h5py.File(load_files+'../src/POMFinder_443structures_100Dataset_per_Structure_xPDF_hypercube_sampling_Grmax_Name.h5', "r")
    y = hf_name.get('y')
    enc = OrdinalEncoder()
    y_onehotenc_cat = enc.fit(np.array(y))
    y_onehotenc_values = enc.fit_transform(np.array(y))

    # Import XYZFinder
    POMFinder = xgb.XGBClassifier()
    POMFinder.load_model(cwd+'../src/XGBoost_443structures_100PDFperStructure_xPDF_hypercube_sampling_Grmax.model')
    return y, y_onehotenc_cat, y_onehotenc_values, POMFinder


def PDF_Preparation(Your_PDF_Name, Qmin, Qmax, Qdamp, rmax=30, nyquist="No", plot=True):
    cwd = os.getcwd()
    for i in range(1000):
        with open(cwd+"/" + Your_PDF_Name, "r") as file:
            data = file.read().splitlines(True)
            if len(data[0]) == 0:
                with open(cwd+"/" + Your_PDF_Name, 'w') as fout:
                    fout.writelines(data[1:])
                break
            first_line = data[0]
            if len(first_line) > 3 and re.match(r'^-?\d+(?:\.\d+)?$', first_line[0]) != None and re.match(r'^-?\d+(?:\.\d+)?$', first_line[1]) == None and re.match(r'^-?\d+(?:\.\d+)?$', first_line[2]) != None:
                PDF = np.loadtxt(cwd+"/" + Your_PDF_Name)
                break
            else:
                with open(cwd+"/" + Your_PDF_Name, 'w') as fout:
                    fout.writelines(data[1:])
        
    #PDF = np.loadtxt("Experimental_Data/" + Your_PDF_Name)
    r, Gr = PDF[:,0], PDF[:,1]
    if r[0] != 0: # In the case that the Data not start at 0.
      Gr = Gr[np.where(r==1)[0][0]:] # Remove Data from 0 to 0.5 AA
      Gr = Gr[::10] # Nyquist sample the rest of the Data
      Gr = np.concatenate(([0,0,0,0,0,0,0,0,0,0], Gr), axis=0) # Concatenate 0 - 0.5 AA on the Gr.
    if nyquist == "No":
      Gr = Gr[::10] # Nyquist sample Data
    if len(Gr) >= (rmax*10+1):
      Gr = Gr[:(rmax*10+1)] # In the case Data is up to more than 30 AA, we do not use it.
    else:
      Gr = np.concatenate((Gr, np.zeros((101-len(Gr),))), axis=0) # In case Data is not going to 30 AA, we add 0's.

    Gr[:10] = np.zeros((10,))
    r = np.arange(0, (rmax+0.1), 0.1)
    # Normalise it to the data from the database
    #Gr += np.abs(np.min(Gr))
    Gr /= np.max(Gr)
    # Add experimental parameters to the Gr
    Gr = np.expand_dims(np.concatenate((np.expand_dims(Qmin, axis=0), np.expand_dims(Qmax, axis=0), np.expand_dims(Qdamp, axis=0), Gr), axis=0), axis=0)

    if plot:
        # Plot the transformation to make sure everything is alright
        plt.plot(PDF[:,0], PDF[:,1], label="Original Data")
        plt.plot(r, Gr[0,3:], label="Gr ready for ML")
        plt.legend()
        plt.title("Original Data vs. transformed Data")
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
    print ("The 1st guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[0]])[2:-2]+"cale.xyz", "with ", y_pred_proba[res[0]]*100, "% certaincy")
    print ("The 2nd guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[1]])[2:-2]+"cale.xyz", "with ", y_pred_proba[res[1]]*100, "% certaincy")
    print ("The 3rd guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[2]])[2:-2]+"cale.xyz", "with ", y_pred_proba[res[2]]*100, "% certaincy")
    print ("The 4th guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[3]])[2:-2]+"cale.xyz", "with ", y_pred_proba[res[3]]*100, "% certaincy")
    print ("The 5th guess from the model is: ", str(y_onehotenc_cat.categories_[0][res[4]])[2:-2]+"cale.xyz", "with ", y_pred_proba[res[4]]*100, "% certaincy")
    
    return res, y_pred_proba


def plot_simulated_predictions(Gr, res, y_onehotenc_cat, Qmin, Qmax, Qdamp):
    from bokeh.io import output_notebook, show
    from bokeh.plotting import figure
    from bokeh.layouts import gridplot
    PDFcalc = DebyePDFCalculator(rmin=0, rmax=30.1, rstep=0.1, qmin=Qmin, qmax=Qmax, qdamp=Qdamp)
    if (str(y_onehotenc_cat.categories_[0][res[0]])[2:6]) == "icsd":
        POMFinder_prediction_1 = str(y_onehotenc_cat.categories_[0][res[0]])[2:-2]+"cale.xyz"
    else:
        POMFinder_prediction_1 = str(y_onehotenc_cat.categories_[0][res[0]])[2:-2]+".xyz"    
    if (str(y_onehotenc_cat.categories_[0][res[1]])[2:6]) == "icsd":
        POMFinder_prediction_2 = str(y_onehotenc_cat.categories_[0][res[1]])[2:-2]+"cale.xyz"
    else:
        POMFinder_prediction_2 = str(y_onehotenc_cat.categories_[0][res[1]])[2:-2]+".xyz"
    if (str(y_onehotenc_cat.categories_[0][res[2]])[2:6]) == "icsd":
        POMFinder_prediction_3 = str(y_onehotenc_cat.categories_[0][res[2]])[2:-2]+"cale.xyz"
    else:
        POMFinder_prediction_3 = str(y_onehotenc_cat.categories_[0][res[2]])[2:-2]+".xyz"
    if (str(y_onehotenc_cat.categories_[0][res[3]])[2:6]) == "icsd":
        POMFinder_prediction_4 = str(y_onehotenc_cat.categories_[0][res[3]])[2:-2]+"cale.xyz"
    else:
        POMFinder_prediction_4 = str(y_onehotenc_cat.categories_[0][res[3]])[2:-2]+".xyz"
    if (str(y_onehotenc_cat.categories_[0][res[4]])[2:6]) == "icsd":
        POMFinder_prediction_5 = str(y_onehotenc_cat.categories_[0][res[4]])[2:-2]+"cale.xyz"        
    else:
        POMFinder_prediction_5 = str(y_onehotenc_cat.categories_[0][res[4]])[2:-2]+".xyz"        

    r, g1 = PDFcalc(loadStructure(cwd+"/Backend/Database/COD_ICSD_XYZs_POMs_unique99/"+POMFinder_prediction_1))
    r, g2 = PDFcalc(loadStructure(cwd+"/Backend/Database/COD_ICSD_XYZs_POMs_unique99/"+POMFinder_prediction_2))
    r, g3 = PDFcalc(loadStructure(cwd+"/Backend/Database/COD_ICSD_XYZs_POMs_unique99/"+POMFinder_prediction_3))
    r, g4 = PDFcalc(loadStructure(cwd+"/Backend/Database/COD_ICSD_XYZs_POMs_unique99/"+POMFinder_prediction_4))
    r, g5 = PDFcalc(loadStructure(cwd+"/Backend/Database/COD_ICSD_XYZs_POMs_unique99/"+POMFinder_prediction_5))
    g1[:10] = np.zeros((10,))
    g2[:10] = np.zeros((10,))
    g3[:10] = np.zeros((10,))
    g4[:10] = np.zeros((10,))
    g5[:10] = np.zeros((10,))
    #g1 -= np.min(g1)
    #g2 -= np.min(g2)
    #g3 -= np.min(g3)
    #g4 -= np.min(g4)
    #g5 -= np.min(g5)
    g1 /= np.max(g1)
    g2 /= np.max(g2)
    g3 /= np.max(g3)
    g4 /= np.max(g4)
    g5 /= np.max(g5)

    output_notebook()
    tools = "hover, box_zoom, undo, crosshair"
    p1 = figure(tools=tools, plot_width = 250, plot_height=350, background_fill_color="silver")
    p2 = figure(tools=tools, plot_width = 250, plot_height=350, background_fill_color="silver")
    p3 = figure(tools=tools, plot_width = 250, plot_height=350, background_fill_color="silver")
    p4 = figure(tools=tools, plot_width = 250, plot_height=350, background_fill_color="silver")
    p5 = figure(tools=tools, plot_width = 250, plot_height=350, background_fill_color="silver")
    
    p1.line(r, g1, legend_label="Simulated PDF: " + POMFinder_prediction_1[:-4], color="red")
    p1.line(r, Gr[0,3:], legend_label="Experimental PDF", color="blue")    
    p2.line(r, g2, legend_label="Simulated PDF: " + POMFinder_prediction_2[:-4], color="red")
    p2.line(r, Gr[0,3:], legend_label="Experimental PDF", color="blue")    
    p3.line(r, g3, legend_label="Simulated PDF: " + POMFinder_prediction_3[:-4], color="red")
    p3.line(r, Gr[0,3:], legend_label="Experimental PDF", color="blue")    
    p4.line(r, g4, legend_label="Simulated PDF: " + POMFinder_prediction_4[:-4], color="red")
    p4.line(r, Gr[0,3:], legend_label="Experimental PDF", color="blue")    
    p5.line(r, g5, legend_label="Simulated PDF: " + POMFinder_prediction_5[:-4], color="red")
    p5.line(r, Gr[0,3:], legend_label="Experimental PDF", color="blue")    

    p1.legend.label_text_font_size = '8pt'
    p1.legend.location = "bottom_right"
    p1.legend.background_fill_alpha = 0.3
    p2.legend.label_text_font_size = '8pt'
    p2.legend.location = "bottom_right"
    p2.legend.background_fill_alpha = 0.3
    p3.legend.label_text_font_size = '8pt'
    p3.legend.location = "bottom_right"
    p3.legend.background_fill_alpha = 0.3
    p4.legend.label_text_font_size = '8pt'
    p4.legend.location = "bottom_right"
    p4.legend.background_fill_alpha = 0.3
    p5.legend.label_text_font_size = '8pt'
    p5.legend.location = "bottom_right"
    p5.legend.background_fill_alpha = 0.3
    grid = gridplot([[p1, p2, p3], [p4, p5]], plot_width=330, plot_height=250)
    show(grid)
    
    return r, Gr[0,3:], g1, g2, g3, g4, g5

