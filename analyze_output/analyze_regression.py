from analyze_output.analyze_ml_regression import analyze_ml_regression
from analyze_output.analyze_fitqun_regression import analyze_fitqun_regression

from analyze_output.analyze_mcData import analyze_ml_regression_dataMC, analyze_fitqun_regression_dataMC

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

import numpy as np


class analysisResults():
     def __init__(self,settings):
        self.settings=settings
        self.global_perf = {}
        self.var_perf = {}
        self.output_value = {}
     def add_global_perf(self, name, axis, quantile, quantile_error, median, median_error):
        self.global_perf[name+self.settings.plotName] = (axis, [quantile, quantile_error, median, median_error])
     def get_global_perf(self, name, axis=None, value=None):
        if value is not None and axis is not None:
            axis_idx = self.global_perf[name+self.settings.plotName][0].index(axis)
            return self.global_perf[name+self.settings.plotName][1][self._global_value_idx(value)][axis_idx]
        elif value is not None:
           return self.global_perf[name+self.settings.plotName][1][self._global_value_idx(value)] 
        elif axis is not None:
            axis_idx = self.global_perf[name+self.settings.plotName][0].index(axis)
            return self.global_perf[name+self.settings.plotName][1][:,axis_idx]
        else:
            return self.global_perf[name+self.settings.plotName]
     def _global_value_idx(self, value):
        if "quantile" in value and "error" in value:
            return 1
        elif "quantile" in value:
            return 0
        elif "median" in value and "error" in value:
            return 2
        elif "median" in value:
            return 3
     def print_global_perf(self):
         print(f"Global perf: {self.global_perf}")

     def add_var_perf(self, name, dictionary):
            self.var_perf[name+self.settings.plotName] =  dictionary
     def get_var_perf(self, name, variable=None, axis=None, value=None):
        if value is not None and axis is not None:
            return np.array(self.var_perf[name+self.settings.plotName][variable][self._var_value_idx(value)][axis])
        elif value is not None:
           return self.var_perf[name+self.settings.plotName][variable][self._var_value_idx(value)] 
        else:
            return self.var_perf[name+self.settings.plotName][variable]

     def add_output_value(self, name, dictionary):
            self.output_value[name+self.settings.plotName] =  dictionary
     def get_output_value(self, name, variable=None, axis=None, value=None):
        if value is not None and axis is not None:
            return np.array(self.output_value[name+self.settings.plotName][variable][self._output_value_idx(value)][axis])
        elif value is not None:
           return self.output_value[name+self.settings.plotName][variable][self._output_value_idx(value)] 
        else:
            return self.output_value[name+self.settings.plotName][variable]


     def _var_value_idx(self, value):
        if "bins" in value:
            return 0
        if "quantile" in value and "error" in value:
            return 2
        elif "quantile" in value:
            return 1
        elif "median" in value and "error" in value:
            return 4
        elif "median" in value:
            return 3

     def _output_value_idx(self, value):
        if "bins" in value:
            return 0
        if "pred" in value and "error" in value:
            return 2
        elif "pred" in value:
            return 1
        elif "truth" in value and "error" in value:
            return 4
        elif "truth" in value:
            return 3
     

def analyze_regression(settings):

    results = analysisResults(settings)

    if settings.doDataMC:
        total_charge_cut = [1000,15000]
        nhits_cut = 200
        if settings.doFiTQun:
            analyze_fitqun_regression_dataMC(settings, total_charge_cut, nhits_cut)
        if settings.doML:
            analyze_ml_regression_dataMC(settings, total_charge_cut, nhits_cut)

    else:
        if settings.doFiTQun:

            #vertex_axis_fq, quantile_lst_fq, quantile_error_lst_fq, median_lst_fq, median_error_lst_fq = analyze_fitqun_regression(settings)
            single_fq_analysis, multi_fq_analysis, output_fq_analysis, dir_f, tw_f = analyze_fitqun_regression(settings) 
            results.add_global_perf("fitqun", single_fq_analysis[0], single_fq_analysis[1], single_fq_analysis[2], single_fq_analysis[3], single_fq_analysis[4])
            results.add_var_perf("fitqun",multi_fq_analysis)
            results.add_output_value("fitqun",output_fq_analysis)
            #results.add_global_perf("fitqun", vertex_axis_fq, quantile_lst_fq, quantile_error_lst_fq, median_lst_fq, median_error_lst_fq)

        if settings.doML:
            #vertex_axis_ml, quantile_lst_ml, quantile_error_lst_ml, median_lst_ml, median_error_lst_ml = analyze_ml_regression(settings)
            print(settings.particleLabel)
            print(settings.inputPath)
            print(settings.fitqunPath)
            print(settings.mlPath)
            print(settings.target)
            single_ml_analysis, multi_ml_analysis, output_ml_analysis = analyze_ml_regression(settings, dir_f, tw_f) 
            print(f"SINGLE ANALYSIS: {single_ml_analysis}")
            #print(f"MULTI ANALYSIS: {multi_ml_analysis}")
            results.add_global_perf("ML", single_ml_analysis[0], single_ml_analysis[1], single_ml_analysis[2], single_ml_analysis[3], single_ml_analysis[4])
            results.add_var_perf("ML",multi_ml_analysis)
            results.add_output_value("ML",output_ml_analysis)



    #print(f"Quantile: {results.get_var_perf('ML', variable='ve', axis='Angle', value='quantile')}")
    #print(f"Median: {results.get_var_perf('ML', variable='ve', axis='Angle', value='median')}")

    
    #print(results.get_global_perf("fitqun", axis="Angle", value="quantile"))
    #print(results.get_global_perf("ML", axis="Angle", value="quantile"))

    if "directions" in settings.target and not settings.doDataMC:
        if settings.doCombination or (settings.doML and settings.doFiTQun):
            print(f"Directions; ML; Angle")
            print(f"Resolution {results.get_global_perf('ML', axis='Angle', value='quantile')} ({results.get_global_perf('ML', axis='Angle', value='quantile error')})")
            print(f"Bias {results.get_global_perf('ML', axis='Angle', value='median')} ({results.get_global_perf('ML', axis='Angle', value='median error')})")
            print(f"Directions; fiTQun; Angle")
            print(f"Resolution {results.get_global_perf('fitqun', axis='Angle', value='quantile')} ({results.get_global_perf('fitqun', axis='Angle', value='quantile error')})")
            print(f"Bias {results.get_global_perf('fitqun', axis='Angle', value='median')} ({results.get_global_perf('fitqun', axis='Angle', value='median error')})")
            if settings.doVarPlots:
                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Angle', value='bins'), results.get_var_perf("ML", variable='ve', axis='Angle', value='quantile'),
                                results.get_var_perf("ML", variable='ve', axis='Angle', value='quantile error'), results.get_var_perf("fitqun", variable='ve', axis='Angle', value='quantile'),
                                results.get_var_perf("fitqun", variable='ve', axis='Angle', value='quantile error'), "Visible Energy [MeV]", "Angle Resolution [deg]", "ve_angle_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Angle', value='bins'), results.get_var_perf("ML", variable='towall', axis='Angle', value='quantile'),
                                results.get_var_perf("ML", variable='towall', axis='Angle', value='quantile error'), results.get_var_perf("fitqun", variable='towall', axis='Angle', value='quantile'),
                                results.get_var_perf("fitqun", variable='towall', axis='Angle', value='quantile error'), "Towall [cm]", "Angle Resolution [deg]", "towall_angle_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Angle', value='bins'), results.get_var_perf("ML", variable='ve', axis='Angle', value='median'),
                                results.get_var_perf("ML", variable='ve', axis='Angle', value='median error'), results.get_var_perf("fitqun", variable='ve', axis='Angle', value='median'),
                                results.get_var_perf("fitqun", variable='ve', axis='Angle', value='median error'), "Visible Energy [MeV]", "Angle Bias [deg]", "ve_angle_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Angle', value='bins'), results.get_var_perf("ML", variable='towall', axis='Angle', value='median'),
                                results.get_var_perf("ML", variable='towall', axis='Angle', value='median error'), results.get_var_perf("fitqun", variable='towall', axis='Angle', value='median'),
                            results.get_var_perf("fitqun", variable='towall', axis='Angle', value='median error'), "Towall [cm]", "Angle Bias [deg]", "towall_angle_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Angle', value='bins'), results.get_var_perf("ML", variable='zenith', axis='Angle', value='quantile'),
                                results.get_var_perf("ML", variable='zenith', axis='Angle', value='quantile error'), results.get_var_perf("fitqun", variable='zenith', axis='Angle', value='quantile'),
                                results.get_var_perf("fitqun", variable='zenith', axis='Angle', value='quantile error'), "cos(zenith)", "Direction Resolution [deg]", "zenith_angle_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Angle', value='bins'), results.get_var_perf("ML", variable='zenith', axis='Angle', value='median'),
                                results.get_var_perf("ML", variable='zenith', axis='Angle', value='median error'), results.get_var_perf("fitqun", variable='zenith', axis='Angle', value='median'),
                                results.get_var_perf("fitqun", variable='zenith', axis='Angle', value='median error'), "cos(zenith)", "Direction Bias [deg]", "zenith_angle_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Angle', value='bins'), results.get_var_perf("ML", variable='azimuth', axis='Angle', value='quantile'),
                                results.get_var_perf("ML", variable='azimuth', axis='Angle', value='quantile error'), results.get_var_perf("fitqun", variable='azimuth', axis='Angle', value='quantile'),
                                results.get_var_perf("fitqun", variable='azimuth', axis='Angle', value='quantile error'), "azimuth [deg]", "Direction Resolution [deg]", "azimuth_angle_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Angle', value='bins'), results.get_var_perf("ML", variable='azimuth', axis='Angle', value='median'),
                                results.get_var_perf("ML", variable='azimuth', axis='Angle', value='median error'), results.get_var_perf("fitqun", variable='azimuth', axis='Angle', value='median'),
                                results.get_var_perf("fitqun", variable='azimuth', axis='Angle', value='median error'), "azimuth [deg]", "Direction Bias [deg]", "azimuth_angle_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Angle', value='bins'), results.get_var_perf("ML", variable='dwall', axis='Angle', value='quantile'),
                                results.get_var_perf("ML", variable='dwall', axis='Angle', value='quantile error'), results.get_var_perf("fitqun", variable='dwall', axis='Angle', value='quantile'),
                                results.get_var_perf("fitqun", variable='dwall', axis='Angle', value='quantile error'), "dwall [cm]", "Direction Resolution [deg]", "dwall_angle_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Angle', value='bins'), results.get_var_perf("ML", variable='dwall', axis='Angle', value='median'),
                                results.get_var_perf("ML", variable='dwall', axis='Angle', value='median error'), results.get_var_perf("fitqun", variable='dwall', axis='Angle', value='median'),
                                results.get_var_perf("fitqun", variable='dwall', axis='Angle', value='median error'), "dwall [cm]", "Direction Bias [deg]", "dwall_angle_ml_fq_median", settings)


        elif settings.doML:
            print(f"Directions; ML; Angle")
            print(f"Resolution {results.get_global_perf('ML', axis='Angle', value='quantile')} ({results.get_global_perf('ML', axis='Angle', value='quantile error')})")
            print(f"Bias {results.get_global_perf('ML', axis='Angle', value='median')} ({results.get_global_perf('ML', axis='Angle', value='median error')})")
            if settings.doVarPlots:
                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Angle', value='bins'), results.get_var_perf("ML", variable='ve', axis='Angle', value='quantile'),
                                results.get_var_perf("ML", variable='ve', axis='Angle', value='quantile error'), "Visible Energy [MeV]", "Angle Resolution [deg]", "ve_angle_ml_fq_res", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Angle', value='bins'), results.get_var_perf("ML", variable='towall', axis='Angle', value='quantile'),
                                results.get_var_perf("ML", variable='towall', axis='Angle', value='quantile error'), "Towall [cm]", "Angle Resolution [deg]", "towall_angle_ml_fq_res", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Angle', value='bins'), results.get_var_perf("ML", variable='ve', axis='Angle', value='median'),
                                results.get_var_perf("ML", variable='ve', axis='Angle', value='median error'),
                                "Visible Energy [MeV]", "Angle Bias [deg]", "ve_angle_ml_fq_median", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Angle', value='bins'), results.get_var_perf("ML", variable='towall', axis='Angle', value='median'),
                                results.get_var_perf("ML", variable='towall', axis='Angle', value='median error'), 
                                "Towall [cm]", "Angle Bias [deg]", "towall_angle_ml_fq_median", settings)

    if "momenta" in settings.target and not settings.doDataMC:
        if settings.doCombination or (settings.doML and settings.doFiTQun):
            print(f"Momenta; ML; Global")
            print(f"Resolution {100*results.get_global_perf('ML', axis='Global', value='quantile')} ({100*results.get_global_perf('ML', axis='Global', value='quantile error')})")
            print(f"Bias {100*results.get_global_perf('ML', axis='Global', value='median')} ({100*results.get_global_perf('ML', axis='Global', value='median error')})")
            print(f"Momenta; fiTQun; Global")
            print(f"Resolution {100*results.get_global_perf('fitqun', axis='Global', value='quantile')} ({100*results.get_global_perf('fitqun', axis='Global', value='quantile error')})")
            print(f"Bias {100*results.get_global_perf('fitqun', axis='Global', value='median')} ({100*results.get_global_perf('fitqun', axis='Global', value='median error')})")
            if settings.doVarPlots:
                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='ve', axis='Global', value='quantile'),
                                100*results.get_var_perf("ML", variable='ve', axis='Global', value='quantile error'), 100*results.get_var_perf("fitqun", variable='ve', axis='Global', value='quantile'),
                                100*results.get_var_perf("fitqun", variable='ve', axis='Global', value='quantile error'), "Visible Energy [MeV]", " Momentum Resolution (%)", "ve_mom_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='towall', axis='Global', value='quantile'),
                                100*results.get_var_perf("ML", variable='towall', axis='Global', value='quantile error'), 100*results.get_var_perf("fitqun", variable='towall', axis='Global', value='quantile'),
                                100*results.get_var_perf("fitqun", variable='towall', axis='Global', value='quantile error'), "Towall [cm]", "Momentum Resolution [%]", "towall_mom_ml_fq_res", settings)


                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='ve', axis='Global', value='median'),
                                100*results.get_var_perf("ML", variable='ve', axis='Global', value='median error'), 100*results.get_var_perf("fitqun", variable='ve', axis='Global', value='median'),
                                100*results.get_var_perf("fitqun", variable='ve', axis='Global', value='median error'), "Visible Energy [MeV]", "Momentum Bias [%]", "ve_mom_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='towall', axis='Global', value='median'),
                                100*results.get_var_perf("ML", variable='towall', axis='Global', value='median error'), 100*results.get_var_perf("fitqun", variable='towall', axis='Global', value='median'),
                                100*results.get_var_perf("fitqun", variable='towall', axis='Global', value='median error'), "Towall [cm]", "Momentum Bias [%]", "towall_mom_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='zenith', axis='Global', value='quantile'),
                                100*results.get_var_perf("ML", variable='zenith', axis='Global', value='quantile error'), 100*results.get_var_perf("fitqun", variable='zenith', axis='Global', value='quantile'),
                                100*results.get_var_perf("fitqun", variable='zenith', axis='Global', value='quantile error'), "zenith [cm]", "Momentum Resolution [%]", "zenith_mom_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='zenith', axis='Global', value='median'),
                                100*results.get_var_perf("ML", variable='zenith', axis='Global', value='median error'), 100*results.get_var_perf("fitqun", variable='zenith', axis='Global', value='median'),
                                100*results.get_var_perf("fitqun", variable='zenith', axis='Global', value='median error'), "zenith [cm]", "Momentum Bias [%]", "zenith_mom_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='azimuth', axis='Global', value='quantile'),
                                100*results.get_var_perf("ML", variable='azimuth', axis='Global', value='quantile error'), 100*results.get_var_perf("fitqun", variable='azimuth', axis='Global', value='quantile'),
                                100*results.get_var_perf("fitqun", variable='azimuth', axis='Global', value='quantile error'), "azimuth [cm]", "Momentum Resolution [%]", "azimuth_mom_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='azimuth', axis='Global', value='median'),
                                100*results.get_var_perf("ML", variable='azimuth', axis='Global', value='median error'), 100*results.get_var_perf("fitqun", variable='azimuth', axis='Global', value='median'),
                                100*results.get_var_perf("fitqun", variable='azimuth', axis='Global', value='median error'), "azimuth [cm]", "Momentum Bias [%]", "azimuth_mom_ml_fq_median", settings)

                if not settings.getfiTQunTruth:
                        plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='dwall', axis='Global', value='quantile'),
                                        100*results.get_var_perf("ML", variable='dwall', axis='Global', value='quantile error'), 100*results.get_var_perf("fitqun", variable='dwall', axis='Global', value='quantile'),
                                        100*results.get_var_perf("fitqun", variable='dwall', axis='Global', value='quantile error'), "dwall [cm]", "Momentum Resolution [%]", "dwall_mom_ml_fq_res", settings)
                                        
                        plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='dwall', axis='Global', value='median'),
                                        100*results.get_var_perf("ML", variable='dwall', axis='Global', value='median error'), 100*results.get_var_perf("fitqun", variable='dwall', axis='Global', value='median'),
                                        100*results.get_var_perf("fitqun", variable='dwall', axis='Global', value='median error'), "dwall [cm]", "Momentum Bias [%]", "dwall_mom_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='z', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='z', axis='Global', value='quantile'),
                                100*results.get_var_perf("ML", variable='z', axis='Global', value='quantile error'), 100*results.get_var_perf("fitqun", variable='z', axis='Global', value='quantile'),
                                100*results.get_var_perf("fitqun", variable='z', axis='Global', value='quantile error'), "z [cm]", "Momentum Resolution [%]", "z_mom_ml_fq_res", settings)
                                
                plot_reg_results(results.get_var_perf("ML", variable='z', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='z', axis='Global', value='median'),
                                100*results.get_var_perf("ML", variable='z', axis='Global', value='median error'), 100*results.get_var_perf("fitqun", variable='z', axis='Global', value='median'),
                                100*results.get_var_perf("fitqun", variable='z', axis='Global', value='median error'), "z [cm]", "Momentum Bias [%]", "z_mom_ml_fq_median", settings)
        elif settings.doML:
            print(f"Directions; ML; Global")
            print(f"Resolution {100*results.get_global_perf('ML', axis='Global', value='quantile')} ({100*results.get_global_perf('ML', axis='Global', value='quantile error')})")
            print(f"Bias {100*results.get_global_perf('ML', axis='Global', value='median')} ({100*results.get_global_perf('ML', axis='Global', value='median error')})")
            if settings.doVarPlots:
                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='ve', axis='Global', value='quantile'),
                                100*results.get_var_perf("ML", variable='ve', axis='Global', value='quantile error'), "Visible Energy [MeV]", " Momentum Resolution (%)", "ve_mom_ml_fq_res", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='towall', axis='Global', value='quantile'),
                                100*results.get_var_perf("ML", variable='towall', axis='Global', value='quantile error'), "Towall [cm]", "Momentum Resolution [%]", "towall_mom_ml_fq_res", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='ve', axis='Global', value='median'),
                                100*results.get_var_perf("ML", variable='ve', axis='Global', value='median error'), "Visible Energy [MeV]", "Momentum Bias [%]", "ve_mom_ml_fq_median", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), 100*results.get_var_perf("ML", variable='towall', axis='Global', value='median'),
                                100*results.get_var_perf("ML", variable='towall', axis='Global', value='median error'), "Towall [cm]", "Momentum Bias [%]", "towall_mom_ml_fq_median", settings)

    if "positions" in settings.target and not settings.doDataMC:

        if settings.doCombination or (settings.doML and settings.doFiTQun):
            if settings.doOutputPlots:
                plot_reg_results(results.get_output_value("ML", variable='X', axis='Global', value='bins'), results.get_output_value("ML", variable='X', axis='Global', value='pred'),
                                results.get_output_value("ML", variable='X', axis='Global', value='pred error'), results.get_output_value("fitqun", variable='X', axis='Global', value='pred'),
                                results.get_output_value("fitqun", variable='X', axis='Global', value='pred error'), "X [cm]", "Global Resolution [cm]", "X_global_pos_ml_fq", settings)
                plot_reg_results(results.get_output_value("ML", variable='Y', axis='Global', value='bins'), results.get_output_value("ML", variable='Y', axis='Global', value='pred'),
                                results.get_output_value("ML", variable='Y', axis='Global', value='pred error'), results.get_output_value("fitqun", variable='Y', axis='Global', value='pred'),
                                results.get_output_value("fitqun", variable='Y', axis='Global', value='pred error'), "Y [cm]", "Global Resolution [cm]", "Y_global_pos_ml_fq", settings)
                plot_reg_results(results.get_output_value("ML", variable='Z', axis='Global', value='bins'), results.get_output_value("ML", variable='Z', axis='Global', value='pred'),
                                results.get_output_value("ML", variable='Z', axis='Global', value='pred error'), results.get_output_value("fitqun", variable='Z', axis='Global', value='pred'),
                                results.get_output_value("fitqun", variable='Z', axis='Global', value='pred error'), 
                                "Z [cm]", "Global Resolution [cm]", "Z_global_pos_ml_fq", settings,
                                truth=results.get_output_value("ML", variable='Z', axis='Global', value='truth'),
                                truth_error=results.get_output_value("ML", variable='Z', axis='Global', value='truth error'))
                plot_reg_results(results.get_output_value("ML", variable='dwall', axis='Global', value='bins'), results.get_output_value("ML", variable='dwall', axis='Global', value='pred'),
                                results.get_output_value("ML", variable='dwall', axis='Global', value='pred error'), results.get_output_value("fitqun", variable='dwall', axis='Global', value='pred'),
                                results.get_output_value("fitqun", variable='dwall', axis='Global', value='pred error'), "dwall [cm]", "Global Resolution [cm]", "dwall_global_pos_ml_fq", settings)
                plot_reg_results(results.get_output_value("ML", variable='towall', axis='Global', value='bins'), results.get_output_value("ML", variable='towall', axis='Global', value='pred'),
                                results.get_output_value("ML", variable='towall', axis='Global', value='pred error'), results.get_output_value("fitqun", variable='towall', axis='Global', value='pred'),
                                results.get_output_value("fitqun", variable='towall', axis='Global', value='pred error'), "towall [cm]", "Global Resolution [cm]", "towall_global_pos_ml_fq", settings)
            if settings.doVarPlots:
                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), results.get_var_perf("ML", variable='ve', axis='Global', value='quantile'),
                                results.get_var_perf("ML", variable='ve', axis='Global', value='quantile error'), results.get_var_perf("fitqun", variable='ve', axis='Global', value='quantile'),
                                results.get_var_perf("fitqun", variable='ve', axis='Global', value='quantile error'), "Visible Energy [MeV]", "Global Resolution [cm]", "ve_global_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), results.get_var_perf("ML", variable='towall', axis='Global', value='quantile'),
                                results.get_var_perf("ML", variable='towall', axis='Global', value='quantile error'), results.get_var_perf("fitqun", variable='towall', axis='Global', value='quantile'),
                                results.get_var_perf("fitqun", variable='towall', axis='Global', value='quantile error'), "Towall [cm]", "Global Resolution [cm]", "towall_global_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='ve', axis='Transverse', value='quantile'),
                                results.get_var_perf("ML", variable='ve', axis='Transverse', value='quantile error'), results.get_var_perf("fitqun", variable='ve', axis='Transverse', value='quantile'),
                                results.get_var_perf("fitqun", variable='ve', axis='Transverse', value='quantile error'), "Visible Energy [MeV]", "Transverse Resolution [cm]", "ve_transverse_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='towall', axis='Transverse', value='quantile'),
                                results.get_var_perf("ML", variable='towall', axis='Transverse', value='quantile error'), results.get_var_perf("fitqun", variable='towall', axis='Transverse', value='quantile'),
                                results.get_var_perf("fitqun", variable='towall', axis='Transverse', value='quantile error'), "Towall [cm]", "Transverse Resolution [cm]", "towall_transverse_pos_ml_fq_res", settings)
                                
                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='quantile error'), results.get_var_perf("fitqun", variable='ve', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("fitqun", variable='ve', axis='Longitudinal', value='quantile error'), "Visible Energy [MeV]", "Longitudinal Resolution [cm]", "ve_longitudinal_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='quantile error'), results.get_var_perf("fitqun", variable='towall', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("fitqun", variable='towall', axis='Longitudinal', value='quantile error'), "Towall [cm]", "Longitudinal Resolution [cm]", "towall_longitudinal_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), results.get_var_perf("ML", variable='ve', axis='Global', value='median'),
                                results.get_var_perf("ML", variable='ve', axis='Global', value='median error'), results.get_var_perf("fitqun", variable='ve', axis='Global', value='median'),
                                results.get_var_perf("fitqun", variable='ve', axis='Global', value='median error'), "Visible Energy [MeV]", "Global Bias [cm]", "ve_global_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), results.get_var_perf("ML", variable='towall', axis='Global', value='median'),
                                results.get_var_perf("ML", variable='towall', axis='Global', value='median error'), results.get_var_perf("fitqun", variable='towall', axis='Global', value='median'),
                                results.get_var_perf("fitqun", variable='towall', axis='Global', value='median error'), "Towall [cm]", "Global Bias [cm]", "towall_global_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='ve', axis='Transverse', value='median'),
                                results.get_var_perf("ML", variable='ve', axis='Transverse', value='median error'), results.get_var_perf("fitqun", variable='ve', axis='Transverse', value='median'),
                                results.get_var_perf("fitqun", variable='ve', axis='Transverse', value='median error'), "Visible Energy [MeV]", "Transverse Bias [cm]", "ve_transverse_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='towall', axis='Transverse', value='median'),
                                results.get_var_perf("ML", variable='towall', axis='Transverse', value='median error'), results.get_var_perf("fitqun", variable='towall', axis='Transverse', value='median'),
                                results.get_var_perf("fitqun", variable='towall', axis='Transverse', value='median error'), "Towall [cm]", "Transverse Bias [cm]", "towall_transverse_pos_ml_fq_median", settings)
                                
                plot_reg_results(results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='median'),
                                results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='median error'), results.get_var_perf("fitqun", variable='ve', axis='Longitudinal', value='median'),
                                results.get_var_perf("fitqun", variable='ve', axis='Longitudinal', value='median error'), "Visible Energy [MeV]", "Longitudinal Bias [cm]", "ve_longitudinal_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='median'),
                                results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='median error'), results.get_var_perf("fitqun", variable='towall', axis='Longitudinal', value='median'),
                                results.get_var_perf("fitqun", variable='towall', axis='Longitudinal', value='median error'), "Towall [cm]", "Longitudinal Bias [cm]", "towall_longitudinal_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Global', value='bins'), results.get_var_perf("ML", variable='zenith', axis='Global', value='quantile'),
                                results.get_var_perf("ML", variable='zenith', axis='Global', value='quantile error'), results.get_var_perf("fitqun", variable='zenith', axis='Global', value='quantile'),
                                results.get_var_perf("fitqun", variable='zenith', axis='Global', value='quantile error'), "zenith [cm]", "Position Resolution [cm]", "zenith_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Global', value='bins'), results.get_var_perf("ML", variable='zenith', axis='Global', value='median'),
                                results.get_var_perf("ML", variable='zenith', axis='Global', value='median error'), results.get_var_perf("fitqun", variable='zenith', axis='Global', value='median'),
                                results.get_var_perf("fitqun", variable='zenith', axis='Global', value='median error'), "zenith [cm]", "Position Bias [cm]", "zenith_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Global', value='bins'), results.get_var_perf("ML", variable='azimuth', axis='Global', value='quantile'),
                                results.get_var_perf("ML", variable='azimuth', axis='Global', value='quantile error'), results.get_var_perf("fitqun", variable='azimuth', axis='Global', value='quantile'),
                                results.get_var_perf("fitqun", variable='azimuth', axis='Global', value='quantile error'), "azimuth [cm]", "Position Resolution [cm]", "azimuth_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Global', value='bins'), results.get_var_perf("ML", variable='azimuth', axis='Global', value='median'),
                                results.get_var_perf("ML", variable='azimuth', axis='Global', value='median error'), results.get_var_perf("fitqun", variable='azimuth', axis='Global', value='median'),
                                results.get_var_perf("fitqun", variable='azimuth', axis='Global', value='median error'), "azimuth [cm]", "Position Bias [cm]", "azimuth_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='zenith', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("ML", variable='zenith', axis='Longitudinal', value='quantile error'), results.get_var_perf("fitqun", variable='zenith', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("fitqun", variable='zenith', axis='Longitudinal', value='quantile error'), "zenith [cm]", "Position Resolution [cm]", "zenith_longitudinal_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='zenith', axis='Longitudinal', value='median'),
                                results.get_var_perf("ML", variable='zenith', axis='Longitudinal', value='median error'), results.get_var_perf("fitqun", variable='zenith', axis='Longitudinal', value='median'),
                                results.get_var_perf("fitqun", variable='zenith', axis='Longitudinal', value='median error'), "zenith [cm]", "Position Bias [cm]", "zenith_longitudinal_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='azimuth', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("ML", variable='azimuth', axis='Longitudinal', value='quantile error'), results.get_var_perf("fitqun", variable='azimuth', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("fitqun", variable='azimuth', axis='Longitudinal', value='quantile error'), "azimuth [cm]", "Position Resolution [cm]", "azimuth_longitudinal_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='azimuth', axis='Longitudinal', value='median'),
                                results.get_var_perf("ML", variable='azimuth', axis='Longitudinal', value='median error'), results.get_var_perf("fitqun", variable='azimuth', axis='Longitudinal', value='median'),
                                results.get_var_perf("fitqun", variable='azimuth', axis='Longitudinal', value='median error'), "azimuth [cm]", "Position Bias [cm]", "azimuth_longitudinal_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='zenith', axis='Transverse', value='quantile'),
                                results.get_var_perf("ML", variable='zenith', axis='Transverse', value='quantile error'), results.get_var_perf("fitqun", variable='zenith', axis='Transverse', value='quantile'),
                                results.get_var_perf("fitqun", variable='zenith', axis='Transverse', value='quantile error'), "zenith [cm]", "Position Resolution [cm]", "zenith_transverse_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='zenith', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='zenith', axis='Transverse', value='median'),
                                results.get_var_perf("ML", variable='zenith', axis='Transverse', value='median error'), results.get_var_perf("fitqun", variable='zenith', axis='Transverse', value='median'),
                                results.get_var_perf("fitqun", variable='zenith', axis='Transverse', value='median error'), "zenith [cm]", "Position Bias [cm]", "zenith_transverse_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='azimuth', axis='Transverse', value='quantile'),
                                results.get_var_perf("ML", variable='azimuth', axis='Transverse', value='quantile error'), results.get_var_perf("fitqun", variable='azimuth', axis='Transverse', value='quantile'),
                                results.get_var_perf("fitqun", variable='azimuth', axis='Transverse', value='quantile error'), "azimuth [cm]", "Position Resolution [cm]", "azimuth_transverse_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='azimuth', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='azimuth', axis='Transverse', value='median'),
                                results.get_var_perf("ML", variable='azimuth', axis='Transverse', value='median error'), results.get_var_perf("fitqun", variable='azimuth', axis='Transverse', value='median'),
                                results.get_var_perf("fitqun", variable='azimuth', axis='Transverse', value='median error'), "azimuth [cm]", "Position Bias [cm]", "azimuth_transverse_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Global', value='bins'), results.get_var_perf("ML", variable='dwall', axis='Global', value='quantile'),
                                results.get_var_perf("ML", variable='dwall', axis='Global', value='quantile error'), results.get_var_perf("fitqun", variable='dwall', axis='Global', value='quantile'),
                                results.get_var_perf("fitqun", variable='dwall', axis='Global', value='quantile error'), "dwall [cm]", "Position Resolution [cm]", "dwall_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Global', value='bins'), results.get_var_perf("ML", variable='dwall', axis='Global', value='median'),
                                results.get_var_perf("ML", variable='dwall', axis='Global', value='median error'), results.get_var_perf("fitqun", variable='dwall', axis='Global', value='median'),
                                results.get_var_perf("fitqun", variable='dwall', axis='Global', value='median error'), "dwall [cm]", "Position Bias [cm]", "dwall_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='dwall', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("ML", variable='dwall', axis='Longitudinal', value='quantile error'), results.get_var_perf("fitqun", variable='dwall', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("fitqun", variable='dwall', axis='Longitudinal', value='quantile error'), "dwall [cm]", "Position Resolution [cm]", "dwall_longitudinal_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='dwall', axis='Longitudinal', value='median'),
                                results.get_var_perf("ML", variable='dwall', axis='Longitudinal', value='median error'), results.get_var_perf("fitqun", variable='dwall', axis='Longitudinal', value='median'),
                                results.get_var_perf("fitqun", variable='dwall', axis='Longitudinal', value='median error'), "dwall [cm]", "Position Bias [cm]", "dwall_longitudinal_pos_ml_fq_median", settings)

                plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='dwall', axis='Transverse', value='quantile'),
                                results.get_var_perf("ML", variable='dwall', axis='Transverse', value='quantile error'), results.get_var_perf("fitqun", variable='dwall', axis='Transverse', value='quantile'),
                                results.get_var_perf("fitqun", variable='dwall', axis='Transverse', value='quantile error'), "dwall [cm]", "Position Resolution [cm]", "dwall_transverse_pos_ml_fq_res", settings)

                plot_reg_results(results.get_var_perf("ML", variable='dwall', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='dwall', axis='Transverse', value='median'),
                                results.get_var_perf("ML", variable='dwall', axis='Transverse', value='median error'), results.get_var_perf("fitqun", variable='dwall', axis='Transverse', value='median'),
                                results.get_var_perf("fitqun", variable='dwall', axis='Transverse', value='median error'), "dwall [cm]", "Position Bias [cm]", "dwall_transverse_pos_ml_fq_median", settings)



            print(f"Positions; ML; Transverse")
            print(f"Resolution {results.get_global_perf('ML', axis='Transverse', value='quantile')} ({results.get_global_perf('ML', axis='Transverse', value='quantile error')})")
            print(f"Bias {results.get_global_perf('ML', axis='Global', value='median')} ({results.get_global_perf('ML', axis='Global', value='median error')})")
            print(f"Directions; fiTQun; Global")
            print(f"Resolution {results.get_global_perf('fitqun', axis='Global', value='quantile')} ({results.get_global_perf('fitqun', axis='Global', value='quantile error')})")
            print(f"Bias {results.get_global_perf('fitqun', axis='Global', value='median')} ({results.get_global_perf('fitqun', axis='Global', value='median error')})")

            print(f"Positions; ML; Transverse")
            print(f"Resolution {results.get_global_perf('ML', axis='Transverse', value='quantile')} ({results.get_global_perf('ML', axis='Transverse', value='quantile error')})")
            print(f"Bias {results.get_global_perf('ML', axis='Transverse', value='median')} ({results.get_global_perf('ML', axis='Transverse', value='median error')})")
            print(f"Directions; fiTQun; Transverse")
            print(f"Resolution {results.get_global_perf('fitqun', axis='Transverse', value='quantile')} ({results.get_global_perf('fitqun', axis='Transverse', value='quantile error')})")
            print(f"Bias {results.get_global_perf('fitqun', axis='Transverse', value='median')} ({results.get_global_perf('fitqun', axis='Transverse', value='median error')})")

            print(f"Positions; ML; Longitudinal")
            print(f"Resolution {results.get_global_perf('ML', axis='Longitudinal', value='quantile')} ({results.get_global_perf('ML', axis='Longitudinal', value='quantile error')})")
            print(f"Bias {results.get_global_perf('ML', axis='Longitudinal', value='median')} ({results.get_global_perf('ML', axis='Longitudinal', value='median error')})")
            print(f"Directions; fiTQun; Longitudinal")
            print(f"Resolution {results.get_global_perf('fitqun', axis='Longitudinal', value='quantile')} ({results.get_global_perf('fitqun', axis='Longitudinal', value='quantile error')})")
            print(f"Bias {results.get_global_perf('fitqun', axis='Longitudinal', value='median')} ({results.get_global_perf('fitqun', axis='Longitudinal', value='median error')})")
        elif settings.doML:
            if settings.doVarPlots:
                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), results.get_var_perf("ML", variable='ve', axis='Global', value='quantile'),
                                results.get_var_perf("ML", variable='ve', axis='Global', value='quantile error'), "Visible Energy [MeV]", "Global Resolution [cm]", "ve_global_pos_ml_fq_res", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), results.get_var_perf("ML", variable='towall', axis='Global', value='quantile'),
                                results.get_var_perf("ML", variable='towall', axis='Global', value='quantile error'), "Towall [cm]", "Global Resolution [cm]", "towall_global_pos_ml_fq_res", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='ve', axis='Transverse', value='quantile'),
                                results.get_var_perf("ML", variable='ve', axis='Transverse', value='quantile error'), "Visible Energy [MeV]", "Transverse Resolution [cm]", "ve_transverse_pos_ml_fq_res", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='towall', axis='Transverse', value='quantile'),
                                results.get_var_perf("ML", variable='towall', axis='Transverse', value='quantile error'), "Towall [cm]", "Transverse Resolution [cm]", "towall_transverse_pos_ml_fq_res", settings)
                                
                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='quantile error'), "Visible Energy [MeV]", "Longitudinal Resolution [cm]", "ve_longitudinal_pos_ml_fq_res", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='quantile'),
                                results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='quantile error'), "Towall [cm]", "Longitudinal Resolution [cm]", "towall_longitudinal_pos_ml_fq_res", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Global', value='bins'), results.get_var_perf("ML", variable='ve', axis='Global', value='median'),
                                results.get_var_perf("ML", variable='ve', axis='Global', value='median error'), "Visible Energy [MeV]", "Global Bias [cm]", "ve_global_pos_ml_fq_median", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Global', value='bins'), results.get_var_perf("ML", variable='towall', axis='Global', value='median'),
                                results.get_var_perf("ML", variable='towall', axis='Global', value='median error'), "Towall [cm]", "Global Bias [cm]", "towall_global_pos_ml_fq_median", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='ve', axis='Transverse', value='median'),
                                results.get_var_perf("ML", variable='ve', axis='Transverse', value='median error'), "Visible Energy [MeV]", "Transverse Bias [cm]", "ve_transverse_pos_ml_fq_median", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Transverse', value='bins'), results.get_var_perf("ML", variable='towall', axis='Transverse', value='median'),
                                results.get_var_perf("ML", variable='towall', axis='Transverse', value='median error'), "Towall [cm]", "Transverse Bias [cm]", "towall_transverse_pos_ml_fq_median", settings)
                                
                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='median'),
                                results.get_var_perf("ML", variable='ve', axis='Longitudinal', value='median error'), "Visible Energy [MeV]", "Longitudinal Bias [cm]", "ve_longitudinal_pos_ml_fq_median", settings)

                plot_mlOnly_reg_results(results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='bins'), results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='median'),
                                results.get_var_perf("ML", variable='towall', axis='Longitudinal', value='median error'), "Towall [cm]", "Longitudinal Bias [cm]", "towall_longitudinal_pos_ml_fq_median", settings)

            print(f"Positions; ML; Global")
            print(f"Resolution {results.get_global_perf('ML', axis='Global', value='quantile')} ({results.get_global_perf('ML', axis='Global', value='quantile error')})")
            print(f"Bias {results.get_global_perf('ML', axis='Global', value='median')} ({results.get_global_perf('ML', axis='Global', value='median error')})")

            print(f"Positions; ML; Transverse")
            print(f"Resolution {results.get_global_perf('ML', axis='Transverse', value='quantile')} ({results.get_global_perf('ML', axis='Transverse', value='quantile error')})")
            print(f"Bias {results.get_global_perf('ML', axis='Transverse', value='median')} ({results.get_global_perf('ML', axis='Transverse', value='median error')})")

            print(f"Positions; ML; Longitudinal")
            print(f"Resolution {results.get_global_perf('ML', axis='Longitudinal', value='quantile')} ({results.get_global_perf('ML', axis='Longitudinal', value='quantile error')})")
            print(f"Bias {results.get_global_perf('ML', axis='Longitudinal', value='median')} ({results.get_global_perf('ML', axis='Longitudinal', value='median error')})")


def plot_reg_results(x, ml, ml_error, fitqun, fitqun_error, xlabel, ylabel, name, settings, truth=None, truth_error=None):
    print(f"length of x: {len(x)}, length of y: {len(ml)}, len of fq: {len(fitqun)}")
    plt.errorbar(x, ml, ml_error, label="ML", ls='none', marker='o')
    plt.errorbar(x, fitqun, fitqun_error, label="fiTQun", ls='none', marker='^')
    if truth is not None and truth_error is not None:
        plt.errorbar(x, truth, truth_error, label="truth", ls='none', marker='+')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(settings.outputPlotPath+'/'+name+'.png', bbox_inches='tight')
    plt.clf()

def plot_mlOnly_reg_results(x, ml, ml_error, xlabel, ylabel, name, settings):
    plt.errorbar(x, ml, ml_error, label="ML", ls='none', marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(settings.outputPlotPath+'/'+name+'.png', bbox_inches='tight')
    plt.clf()

    