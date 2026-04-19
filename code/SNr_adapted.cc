#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
// added to include normal gaussian # generator
#include <random>
#include <iterator>
#include <functional>

// For gaussian number stuff
std::default_random_engine dist_gen(149084257);

// Equation for gate variables:
inline double x_inf(double v, double V12, double k) { return 1. / (1. + exp(-(v - V12) / k)); }
inline double x_inf2(double v, double V12, double k, double min) { return (1. - min) / (1. + exp(-(v - V12) / k)); }
inline double tau_inf(double tau, double v, double tauV12, double tauk) { return tau / (cosh((v - tauV12) / tauk)); }
inline double tau_inf2(double tau0, double v, double tau1, double phi, double sigma0, double sigma1) { return tau0 + (tau1 - tau0) / (exp((phi - v) / sigma0) + exp((phi - v) / sigma1)); }
inline double tau_inf3(double tau0, double v, double tau1, double tauV12, double tauk) { return tau0 + tau1 / (1 + exp((v - tauV12) / tauk)); }
inline double Gr(double v) { return 4.34e-5 * exp(-0.0211539274 * v); } // paper has error 4.34e5 should be e-5
inline double Gd1(double v) { return 0.075 + 0.043 * tanh((v + 20) - 20); }
inline double Gv(double v) { return (10.6408 - 14.6408 * exp(-v / 42.7671)) / v; }

const double c_som = 100.0, c_dend = 0.4 * c_som, c_total = c_som + c_dend;

// NaF
const double mV12 = -30.2, mk = 6.2;
const double mtau0 = 0.05, mtau1 = 0.05, mphi = 1, msigma0 = 1, msigma1 = 1;
const double hV12 = -63.3, hk = -8.1;
const double htau0 = 0.59, htau1 = 35.1, hphi = -43, hsigma0 = 10, hsigma1 = -5;
const double sV12 = -30, sk = -0.4;
const double stau0 = 10, stau1 = 50, sphi = -40, ssigma0 = 18.3, ssigma1 = -10, smin = 0.15;

// double gNaF = 35 * c_som;

// NaP
const double napmV12 = -50, napmk = 3;
const double napmtau0 = 0.03, napmtau1 = 0.146, napmphi = -42.6, napmsigma0 = 14.4, napmsigma1 = -14.4, napmmin = 0.0;
const double naphV12 = -57, naphk = -4.0;
const double naphtau0 = 10, naphtau1 = 17, naphphi = -34, naphsigma0 = 26, naphsigma1 = -31.9, naphmin = 0.154;
double NAP_BLK = 1;

// CaH (high voltage activated Ca (parameters are from Elsen 1998)
const double mcV12 = -27.5, mck = 3.0;
const double mctau = 0.5;
const double hcV12 = -52.4, hck = -5.2;
const double hctau = 18;

// CaL (Low voltage activated Ca (parameters are from Elsen 1998)
const double mcl_V12 = -59.15, mcl_k = 2.36;
const double mcl_tau = 3.2;
const double hcl_V12 = -82.4, hcl_k = -5.31;
const double hcl_tau = 11.6;

// Ca Pump
double Caout = 4, tauCa = 250, Cain0 = 0.00000005; // units of Ca are in mM
// double alphaCa = 1e-6 / c_som, // OLD
double alphaCa = 0.925e-7;
double Kpump = 1e-3, Vpump = Kpump / tauCa;
double ry_f = 5. / 1000.; // frequency of spontaneious ca release
double I_ry = 0, tau_ry = 500.0;

// Kdr
const double kmV12 = -26, kmk = 7.8;
const double kmtau0 = 0.1, kmtau1 = 14, kmphi = -26, kmsigma0 = 13, kmsigma1 = -12;
const double khV12 = -20, khk = -10;
const double khtau0 = 5.0, khtau1 = 20, khphi = 0, khsigma0 = 10, khsigma1 = -10, khmin = 0.6;

// CAN
double K_CAN = 1.15 * 1e-6, nc = 3;
double ECAN = -18.0;
double canh_V12 = -95.0, canh_k = -14.0;
double can_tau0 = 300, can_tau1 = 350, can_tau_V12 = -60, can_tau_k = 3.0;

const double hcanC12 = .00015, hcank = -.0002;
const double hcantau = 100;

// SK
double K_SK = 0.0004, nc_SK = 4, tauSK_inf = .1; // Parameters from Xia_1998_Nature 3uM (units of ca in simulations ire in mM)  0.3*1e-6

// TRPC3
const double ETRPC3 = -37;
double TRPC3_BLOCK = 1;

// HCN
const double mhcn_V12 = -76.4, mhcn_k = -3.3;
const double mhcn_tau0 = 0.0, mhcn_tau1 = 3625, mhcn_phi = -76.4, mhcn_sigma0 = 6.56, mhcn_sigma1 = -7.48;

// NMDA(not used in model)
const double eta = 0.33, Mg = 1.3, gammanmda = 0.05;
const double hnmda12 = 0.0004, knmda = -0.00005, hnmda_tau = 500;
double Pnmda = .00;
double ENMDA = -20, gNMDA = 0.000;

// ChR2(not used in model)
double Echr2 = 0;
double gma_chr2 = 0.1, eps1 = 0.8535, eps2 = 0.14;
double sig_ret = 10e-20, Gd2 = 0.05, tau_chr2 = 1.3;

// Arch(not used in model)
double Earch = -145;

// Cl- dynamics for GABA_A
double tauCL = 180, CLin0 = 0.0037, CLout = 120; // CLout = 120e-3; old
double rho_som = 0.5, rho_dend = 1, r_dend = 2e-6, Vol_som = pow((c_som), 3.0 / 2.0);
// old double alphaCL = 8.85e-5 / (rho_som * pow((c_som), 3.0 / 2.0));
// old double alphaCL_dend = 1.77e-11 / (rho_dend * c_dend * r_dend);
double alphaCL = 1.85e-7;
double alphaCL_dend = 2.3e-6;
double tauCL_dend = 18, CLin0_dend = 0.0017;
double CsCd = c_som / (c_som + c_dend); // ratio of C_soma/C_dend
double gGABA_dr = 0;
double gaba_fixed = 0, EGABA_fixed = -64, EGABA_dend_fixed = -80;
double hco3in = 11.8, hco3out = 25.0;		  // OLD hco3in = 11.8e-3, hco3out = 25.0e-3,
double phco3 = 1, pcl = 4;					  // The values for hco3 in and out and phco3 and pcl come from doyon 2016 frontiers in cellular neurosci.	: phco3=1 and pcl=4 also come from doyon 2016 (changed from these parameters on 7/3/2019 inorder to get more rhobust excititory effect)
double EHCO3 = 26.54 * log(hco3in / hco3out); // added negative should be in / out
// double gcltonic = 0.0,gcltonic_dend = 0.0 ; // commented out to have default params matching test run
double gcltonic = 10.0, gcltonic_dend = 4;

double gcltonic0 = 0, gcltonic1 = 0;
double gcltonic_dend0 = 0, gcltonic_dend1 = 0;
double gkcc20 = 0.15, gkcc21 = 0.15;
double gkcc2_dend0 = 0.15, gkcc2_dend1 = 0.15;

// GPe GABA_A stim (gaba_train 0/1=off/on, p_stim=period, gpe_gaba_w = strength
double gaba_train = 0, p_stim = 0, t_max_pstim = 300, stimf_max = 60, stimf = 1.1, gaba_ton = 0, p_ton = 0, gaba_exc_train = 0, p_exc = 0, reset = 0, gpe_gaba_w = 0, dend_gaba_w = 0, gpe_exc_w = 0, vclamp = 0;
double GPe_f_0 = 0, GPe_f = GPe_f_0, F_OFF = 0, D_OFF = 0, str_f = 0.0;
// double GPe_strength = 0.2, Str_strength = 0.4;
double STN_f_0 = 0, STN_f = STN_f_0, STN_strength = 0.25;
// int gpe_poisson = 0;
// int str_poisson = 0;
//////////////////////////////////////////////////////////////
// Somatic and Dendritic facilitation and depression parameters
//////////////////////////////////////////////////////////////
// facilitation
double alpha_F = 0.125, F_0 = 0.145, tau_F = 1000, F_min = 1; // parameters for facilitation from connelly 2010
// decay
double alpha_D = 0.565, D_0 = 1, tau_D = 1000; // parameters for decay: tau is taken from connelly 2010
double D_min = 0.67, pDr = 1 - D_min;		   // proportion able to depress

// PV, LhX6, STN Frequency parameters(not used in model)
double f_pv = 100.0 / 1000.0, f_lhx6 = 100.0 / 1000.0, f_stn = 100.0 / 1000.0;
double dpv = 0.0, dlhx6 = 0.0, dstn = 0.0;
double w_pv = 0.3, w_lhx6 = 0.1, w_stn = 0.03;

// concentrations
double Nain = 15, Naout = 120;
double Kin = 140, Kout = 4;

// g_tonic_STN
double gSTN_ton0 = 0, gSTN_ton1 = 0;

double ENa = 50, EK = -90, EHCN = -45; //-30;

const double Esyn = -64;
const double Eexc = 0.0;
const double Edrive = -10;

double rnd() { return double(rand()) / RAND_MAX; }

// double tauexc = 5;
//  double tausyn = 3;		  // 8;		  // 3
//  double tausyn_dend = 7.2; // 30
double oneway = 0;
double syncstart = 1;

/************************************************************
 NEURON CLASS
 **********************************************************/
class Neuron
{
public:
	int network_size;
	// parameters

	double gCa, gCaL, gNaP, gCAN, gChR2, gArch, gSK, gHCN, gTRPC3, gSD, gKCC2, gKCC2_dend, ID, GCLtonic, GCLtonic_dend, GCLtonicSOMA;
	double E_leak, g_L, gKdr, gNaF;
	double gstn_mu, gstn_theta, gstn_sigma; // variables for OU process for conductances
	double gcl_mu, gcl_theta, gcl_sigma;
	double gcl_mu_dend, gcl_theta_dend, gcl_sigma_dend;
	double soma_current_intensity, dend_current_intensity, soma_current_intensity_theta, dend_current_intensity_theta, soma_current_noise, dend_current_noise, soma_current_noise_mu, dend_current_noise_mu;

	// dynamical variables
	double v, dv, v_dend, dv_dend;
	double m, h, s;	   // INaF
	double napm, naph; // INaP
	double mc, hc;	   // ICaHigh
	double mcl, hcl;   // ICaLow
	double n;		   // IKdr
	double km, kh;	   // INaP
	double mSK;		   //
	double m_hcn;
	double h_can;
	double h_nmda, m_nmda;
	double Cain;
	double t_lastsp;
	double gsyn, gsyn_dend, gexc, ton_dr, ton_dend_dr, g_tonic_stn;
	double tlastsp;
	double Nain;
	double CLin, CLin_dend;
	double ECL, ECL_dend, ECL_SOMdend, ECL_dendSOM, ECa, EGABA, EGABA_dend, ISOMdend, IDENDsom, I_CL_SOMdend, I_CL_dendSOM;
	double hcan;
	double mcan;
	double OP1, OP2, CL1, CL2, Pchr2; // IChR2
	double Carch, Oarch, Iarch;		  // arch channel states
	double Inoise, ICa, INaF, INaP, ITRPC3, ITRPC3_dend, IKdr, ISK, Ih, IL;
	double last_stim, last_ton, last_exc;
	double pv_sp, lhx6_sp, stn_sp, T;
	double PVFREQ, LHX6FREQ, STNFREQ;
	double F_pre, D_pre;
	// double gdendsyn[100], fdendsyn[100], gDENDSYN, fDENDSYN, gsomsyn[100], dsomsyn[100], gSOMSYN, dSOMSYN, gexc_dend[100], gEXCDEND;
	double g_GABA_gpe, g_GABA_str, gSTN, g_GABA_snr, g_GABA_snr_dend;
	double gpe_Freq, str_Freq, gpe_stim, str_stim, gpe_poisson, str_poisson;
	double W_gpe, gpe_start_time, gpe_stim_length, gpe_base_length, gpe_post_length;
	double W_str, str_start_time, str_stim_length, str_base_length, str_post_length;
	double tausyn, tausyn_dend, tauexc;

	int gpe_stim_count, gpe_pulses, gpe_pulse_length;
	double gpe_end_time, gpe_trial_length;
	int gpe_width, GPe_ON, gpe_max_pulses;

	int str_stim_count, str_pulses, str_pulse_length;
	double str_end_time, str_trial_length;
	int str_width, STR_ON, str_max_pulses;

	double IAPP, IAPP_DEND;

	Neuron();
	void init(double, double, double, double, double, double, double, double, double, double, double, double);
	void step(double dt, double t);
	int spike(double vth, double dt, double t);
};

struct connection
{
	int source, target;
	double weight;
};

/************************************************************
 POPULATION CLASS COMPRISED OF NEURONS
 **********************************************************/
class Population
{
public:
	Neuron *net;
	int size;

	connection *w;
	int sex;

	double vth;

	Population(int n, std::string data_directory)
	{
		// std::ifstream param_matrix(data_directory + "/" + "heterogeneity_matrix.txt");
		vth = -35;
		size = n;				// Threshold to consider a spike
		net = new Neuron[size]; // Array of N neurons

		std::ifstream CL_in_D_file(data_directory + "/model_setup/CL_in_D.txt");
		std::ifstream CL_in_S_file(data_directory + "/model_setup/CL_in_S.txt");
		std::ifstream dend_noise_intensity_theta_file(data_directory + "/model_setup/dend_noise_intensity_theta.txt");
		std::ifstream dend_noise_intensity_file(data_directory + "/model_setup/dend_noise_intensity.txt");
		std::ifstream Eleak_mV_file(data_directory + "/model_setup/Eleak_mV.txt");
		std::ifstream gCA_nS_pF_file(data_directory + "/model_setup/gCA_nS_pF.txt");
		std::ifstream gHCN_nS_pF_file(data_directory + "/model_setup/gHCN_nS_pF.txt");
		std::ifstream gKCC2_D_nS_pF_file(data_directory + "/model_setup/gKCC2_D_nS_pF.txt");
		std::ifstream gKCC2_S_nS_pF_file(data_directory + "/model_setup/gKCC2_S_nS_pF.txt");
		std::ifstream gKDR_nS_pF_file(data_directory + "/model_setup/gKDR_nS_pF.txt");
		std::ifstream gL_nS_pF_file(data_directory + "/model_setup/gL_nS_pF.txt");
		std::ifstream gNAF_nS_pF_file(data_directory + "/model_setup/gNAF_nS_pF.txt");
		std::ifstream gNAP_nS_pF_file(data_directory + "/model_setup/gNAP_nS_pF.txt");
		std::ifstream gpe_base_length_file(data_directory + "/model_setup/gpe_base_length.txt");
		std::ifstream gpe_poisson_file(data_directory + "/model_setup/gpe_poisson.txt");
		std::ifstream gpe_post_length_file(data_directory + "/model_setup/gpe_post_length.txt");
		std::ifstream gpe_start_time_file(data_directory + "/model_setup/gpe_start_time.txt");
		std::ifstream gpe_stim_freqs_file(data_directory + "/model_setup/gpe_stim_freqs.txt");
		std::ifstream gpe_stim_length_file(data_directory + "/model_setup/gpe_stim_length.txt");
		std::ifstream gpe_stim_file(data_directory + "/model_setup/gpe_stim.txt");
		std::ifstream gSD_nS_file(data_directory + "/model_setup/gSD_nS.txt");
		std::ifstream gSK_nS_pF_file(data_directory + "/model_setup/gSK_nS_pF.txt");
		std::ifstream gTON_CL_D_MEAN_nS_pF_file(data_directory + "/model_setup/gTON_CL_D_MEAN_nS_pF.txt");
		std::ifstream gTON_CL_D_SIGMA_file(data_directory + "/model_setup/gTON_CL_D_SIGMA.txt");
		std::ifstream gTON_CL_D_THETA_file(data_directory + "/model_setup/gTON_CL_D_THETA.txt");
		std::ifstream gTON_CL_S_MEAN_nS_pF_file(data_directory + "/model_setup/gTON_CL_S_MEAN_nS_pF.txt");
		std::ifstream gTON_CL_S_SIGMA_file(data_directory + "/model_setup/gTON_CL_S_SIGMA.txt");
		std::ifstream gTON_CL_S_THETA_file(data_directory + "/model_setup/gTON_CL_S_THETA.txt");
		std::ifstream gTON_STN_MEAN_nS_pF_file(data_directory + "/model_setup/gTON_STN_MEAN_nS_pF.txt");
		std::ifstream gTON_STN_SIGMA_file(data_directory + "/model_setup/gTON_STN_SIGMA.txt");
		std::ifstream gTON_STN_THETA_file(data_directory + "/model_setup/gTON_STN_THETA.txt");
		std::ifstream gTRPC3_nS_pF_file(data_directory + "/model_setup/gTRPC3_nS_pF.txt");
		std::ifstream Iapp_dend_file(data_directory + "/model_setup/Iapp_dend.txt");
		std::ifstream Iapp_file(data_directory + "/model_setup/Iapp.txt");
		std::ifstream soma_noise_intensity_theta_file(data_directory + "/model_setup/soma_noise_intensity_theta.txt");
		std::ifstream soma_noise_intensity_file(data_directory + "/model_setup/soma_noise_intensity.txt");
		std::ifstream str_base_length_file(data_directory + "/model_setup/str_base_length.txt");
		std::ifstream str_poisson_file(data_directory + "/model_setup/str_poisson.txt");
		std::ifstream str_post_length_file(data_directory + "/model_setup/str_post_length.txt");
		std::ifstream str_start_time_file(data_directory + "/model_setup/str_start_time.txt");
		std::ifstream str_stim_freqs_file(data_directory + "/model_setup/str_stim_freqs.txt");
		std::ifstream str_stim_length_file(data_directory + "/model_setup/str_stim_length.txt");
		std::ifstream str_stim_file(data_directory + "/model_setup/str_stim.txt");
		std::ifstream W_gpe_file(data_directory + "/model_setup/W_gpe.txt");
		std::ifstream W_str_file(data_directory + "/model_setup/W_str.txt");
		std::ifstream tausyn_file(data_directory + "/model_setup/tausyn.txt");
		std::ifstream tausyn_dend_file(data_directory + "/model_setup/tausyn_dend.txt");
		std::ifstream tauexc_file(data_directory + "/model_setup/tauexc.txt");

		double CL_in_D[size];
		double CL_in_S[size];
		double dend_noise_intensity_theta[size];
		double dend_noise_intensity[size];
		double Eleak_mV[size];
		double gCA_nS_pF[size];
		double gHCN_nS_pF[size];
		double gKCC2_D_nS_pF[size];
		double gKCC2_S_nS_pF[size];
		double gKDR_nS_pF[size];
		double gL_nS_pF[size];
		double gNAF_nS_pF[size];
		double gNAP_nS_pF[size];
		double gpe_base_length[size];
		double gpe_poisson[size];
		double gpe_post_length[size];
		double gpe_start_time[size];
		double gpe_stim_freqs[size];
		double gpe_stim_length[size];
		double gpe_stim[size];
		double gSD_nS[size];
		double gSK_nS_pF[size];
		double gTON_CL_D_MEAN_nS_pF[size];
		double gTON_CL_D_SIGMA[size];
		double gTON_CL_D_THETA[size];
		double gTON_CL_S_MEAN_nS_pF[size];
		double gTON_CL_S_SIGMA[size];
		double gTON_CL_S_THETA[size];
		double gTON_STN_MEAN_nS_pF[size];
		double gTON_STN_SIGMA[size];
		double gTON_STN_THETA[size];
		double gTRPC3_nS_pF[size];
		double Iapp_dend[size];
		double Iapp[size];
		double soma_noise_intensity_theta[size];
		double soma_noise_intensity[size];
		double str_base_length[size];
		double str_poisson[size];
		double str_post_length[size];
		double str_start_time[size];
		double str_stim_freqs[size];
		double str_stim_length[size];
		double str_stim[size];
		double W_gpe[size];
		double W_str[size];
		double tausyn[size];
		double tausyn_dend[size];
		double tauexc[size];

		for (size_t i = 0; i < size; ++i)
		{
			CL_in_D_file >> CL_in_D[i];
			CL_in_S_file >> CL_in_S[i];
			dend_noise_intensity_theta_file >> dend_noise_intensity_theta[i];
			dend_noise_intensity_file >> dend_noise_intensity[i];
			Eleak_mV_file >> Eleak_mV[i];
			gCA_nS_pF_file >> gCA_nS_pF[i];
			gHCN_nS_pF_file >> gHCN_nS_pF[i];
			gKCC2_D_nS_pF_file >> gKCC2_D_nS_pF[i];
			gKCC2_S_nS_pF_file >> gKCC2_S_nS_pF[i];
			gKDR_nS_pF_file >> gKDR_nS_pF[i];
			gL_nS_pF_file >> gL_nS_pF[i];
			gNAF_nS_pF_file >> gNAF_nS_pF[i];
			gNAP_nS_pF_file >> gNAP_nS_pF[i];
			gpe_base_length_file >> gpe_base_length[i];
			gpe_poisson_file >> gpe_poisson[i];
			gpe_post_length_file >> gpe_post_length[i];
			gpe_start_time_file >> gpe_start_time[i];
			gpe_stim_freqs_file >> gpe_stim_freqs[i];
			gpe_stim_length_file >> gpe_stim_length[i];
			gpe_stim_file >> gpe_stim[i];
			gSD_nS_file >> gSD_nS[i];
			gSK_nS_pF_file >> gSK_nS_pF[i];
			gTON_CL_D_MEAN_nS_pF_file >> gTON_CL_D_MEAN_nS_pF[i];
			gTON_CL_D_SIGMA_file >> gTON_CL_D_SIGMA[i];
			gTON_CL_D_THETA_file >> gTON_CL_D_THETA[i];
			gTON_CL_S_MEAN_nS_pF_file >> gTON_CL_S_MEAN_nS_pF[i];
			gTON_CL_S_SIGMA_file >> gTON_CL_S_SIGMA[i];
			gTON_CL_S_THETA_file >> gTON_CL_S_THETA[i];
			gTON_STN_MEAN_nS_pF_file >> gTON_STN_MEAN_nS_pF[i];
			gTON_STN_SIGMA_file >> gTON_STN_SIGMA[i];
			gTON_STN_THETA_file >> gTON_STN_THETA[i];
			gTRPC3_nS_pF_file >> gTRPC3_nS_pF[i];
			Iapp_dend_file >> Iapp_dend[i];
			Iapp_file >> Iapp[i];
			soma_noise_intensity_theta_file >> soma_noise_intensity_theta[i];
			soma_noise_intensity_file >> soma_noise_intensity[i];
			str_base_length_file >> str_base_length[i];
			str_poisson_file >> str_poisson[i];
			str_post_length_file >> str_post_length[i];
			str_start_time_file >> str_start_time[i];
			str_stim_freqs_file >> str_stim_freqs[i];
			str_stim_length_file >> str_stim_length[i];
			str_stim_file >> str_stim[i];
			W_gpe_file >> W_gpe[i];
			W_str_file >> W_str[i];
			tausyn_file >> tausyn[i];
			tausyn_dend_file >> tausyn_dend[i];
			tauexc_file >> tauexc[i];
		}

		for (int i = 0; i < size; i++) // Loop through neurons and set each individual cell property
		{

			net[i].network_size = 1 * size;

			net[i].g_tonic_stn = c_dend * gTON_STN_MEAN_nS_pF[i];
			net[i].gstn_mu = net[i].g_tonic_stn;
			net[i].gstn_sigma = gTON_STN_SIGMA[i];
			net[i].gstn_theta = gTON_STN_THETA[i];

			net[i].GCLtonicSOMA = c_som * gTON_CL_S_MEAN_nS_pF[i];
			net[i].gcl_mu = net[i].GCLtonicSOMA;
			net[i].gcl_sigma = gTON_CL_S_SIGMA[i];
			net[i].gcl_theta = gTON_CL_S_THETA[i];
			net[i].GCLtonic = net[i].GCLtonicSOMA;

			net[i].GCLtonic_dend = c_dend * gTON_CL_D_MEAN_nS_pF[i];
			net[i].gcl_mu_dend = net[i].GCLtonic_dend;
			net[i].gcl_sigma_dend = gTON_CL_D_SIGMA[i];
			net[i].gcl_theta_dend = gTON_CL_D_THETA[i];

			net[i].gKCC2 = c_som * gKCC2_S_nS_pF[i];
			net[i].gKCC2_dend = c_dend * gKCC2_D_nS_pF[i];
			net[i].gTRPC3 = c_dend * gTRPC3_nS_pF[i];
			net[i].gHCN = c_som * gHCN_nS_pF[i];
			net[i].gCa = c_som * gCA_nS_pF[i];
			net[i].g_L = c_som * gL_nS_pF[i];
			net[i].gSK = c_som * gSK_nS_pF[i];
			net[i].gNaP = c_som * gNAP_nS_pF[i];
			net[i].gNaF = c_som * gNAF_nS_pF[i];
			net[i].gKdr = c_som * gKDR_nS_pF[i];
			net[i].gSD = gSD_nS[i];

			net[i].soma_current_noise = 0;
			net[i].soma_current_noise_mu = 0;
			net[i].soma_current_intensity = soma_noise_intensity[i];
			net[i].soma_current_intensity_theta = soma_noise_intensity_theta[i];
			net[i].dend_current_noise = 0;
			net[i].dend_current_noise_mu = 0;
			net[i].dend_current_intensity = dend_noise_intensity[i];
			net[i].dend_current_intensity_theta = dend_noise_intensity_theta[i];

			net[i].E_leak = Eleak_mV[i];
			net[i].CLin = CL_in_S[i];
			net[i].CLin_dend = CL_in_D[i];
			net[i].IAPP = Iapp[i];
			net[i].IAPP_DEND = Iapp_dend[i];
			net[i].tausyn = tausyn[i];
			net[i].tauexc = tauexc[i];
			net[i].tausyn_dend = tausyn_dend[i];

			if (gpe_stim[i] > 0)
			{
				net[i].gpe_stim = gpe_stim[i];
				net[i].gpe_Freq = gpe_stim_freqs[i];
				net[i].gpe_poisson = gpe_poisson[i];
				net[i].W_gpe = W_gpe[i];
				net[i].gpe_start_time = gpe_start_time[i] * 1000;
				net[i].gpe_base_length = gpe_base_length[i] * 1000;
				net[i].gpe_stim_length = gpe_stim_length[i] * 1000;
				net[i].gpe_post_length = gpe_post_length[i] * 1000;
				net[i].gpe_stim_count = 0;
				net[i].gpe_end_time = net[i].gpe_start_time + net[i].gpe_stim_length;
				net[i].gpe_trial_length = net[i].gpe_stim_length + net[i].gpe_base_length + net[i].gpe_post_length;
				net[i].gpe_pulses = 0;
				net[i].gpe_max_pulses = (int)(net[i].gpe_stim_length / 1000) / (1 / net[i].gpe_Freq);
				net[i].gpe_pulse_length = (1 / net[i].gpe_Freq) * 1000;

				net[i].gpe_width = 1;
				net[i].GPe_ON = 0;
			}
			else
			{
				net[i].gpe_stim = 0;
				net[i].gpe_Freq = 0;
				net[i].gpe_poisson = 0;
				net[i].W_gpe = 0;
				net[i].gpe_start_time = 0;
				net[i].gpe_base_length = 0;
				net[i].gpe_stim_length = 0;
				net[i].gpe_post_length = 0;
			}

			if (str_stim[i] > 0)
			{
				net[i].str_stim = str_stim[i];
				net[i].str_Freq = str_stim_freqs[i];
				net[i].str_poisson = str_poisson[i];
				net[i].W_str = W_str[i];
				net[i].str_start_time = str_start_time[i] * 1000;
				net[i].str_base_length = str_base_length[i] * 1000;
				net[i].str_stim_length = str_stim_length[i] * 1000;
				net[i].str_post_length = str_post_length[i] * 1000;
				net[i].str_stim_count = 0;
				net[i].str_end_time = net[i].str_start_time + net[i].str_stim_length;
				net[i].str_trial_length = net[i].str_stim_length + net[i].str_base_length + net[i].str_post_length;
				net[i].str_pulses = 0;
				net[i].str_max_pulses = (int)(net[i].str_stim_length / 1000) / (1 / net[i].str_Freq);
				net[i].str_pulse_length = (1 / net[i].str_Freq) * 1000;

				net[i].str_width = 1;
				net[i].STR_ON = 0;
			}
			else
			{
				net[i].str_stim = 0;
				net[i].str_Freq = 0;
				net[i].str_poisson = 0;
				net[i].W_str = 0;
				net[i].str_start_time = 0;
				net[i].str_base_length = 0;
				net[i].str_stim_length = 0;
				net[i].str_post_length = 0;
			}

			// Cell characteristics
			net[i].t_lastsp = 0;
			net[i].ID = i;
		}
		/************************************************************
		CREATE NETWORK CONNECTIONS
		**********************************************************/

		std::ifstream weight_matrix(data_directory + "/" + "weights.txt");
		double weight_values[size][size];
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t j = 0; j < size; ++j)
			{
				weight_matrix >> weight_values[i][j];
			}
		}
		int num_connections[size];
		w = new connection[size * size];   // SNr array to hold connections
		sex = 0;						   // counter to walk through array
		for (int i = 0; i < size; i++)	   // Can consider array as matrix, i represents rows
			for (int j = 0; j < size; j++) // j represents columns
			{
				if (weight_values[i][j] > 0)
				{
					w[sex].source = i;							 // the ith row, as the ith neuron
					w[sex].target = j;							 // the jth column as the jth neuron
					w[sex].weight = c_som * weight_values[i][j]; // c_som * wght0 + c_som * (wght1 - wght0) * rnd();
					num_connections[j] += 1;
					sex++;
				}
			}
	}

	~Population()
	{
		delete w;
		delete net;
	}

	// UPDATE SYNAPTIC CONNECTIONS
	int step(double dt, int *spk, double t)
	{
		int sp = 0;
		for (int i = 0; i < size; i++)
		{
			spk[i] = net[i].spike(vth, dt, t);
			sp += spk[i];
		}
		for (int i = 0; i < sex; i++)
			if (spk[w[i].source])
				net[w[i].target].g_GABA_snr += w[i].weight;
		for (int i = 0; i < size; i++)
			if (spk[i])
				net[i].t_lastsp = 0;
		return sp;
	}
};

// ############### END POPULATION CLASS #################

// INITIAL CONDITIONS FOR NEURON
Neuron::Neuron()
{
	v = -60 + rnd() * 15 * syncstart;
	dv = 0.0;
	v_dend = -60 + rnd() * 2 * syncstart;
	dv_dend = 0.0;
	m = .1;
	h = .9;
	s = .9;
	km = .01;
	kh = .9;
	napm = .01;
	naph = .04;
	n = .9;
	Cain = .00025;
	mc = .001;
	hc = .001;
	mcl = .1;
	hcl = .1;
	m_hcn = 0.01;
	h_can = 0.05;
	h_nmda = 0.99;
	m_nmda = 0;
	CLin = 0.006;
	CLin_dend = 0.006;
	gCa = .01;
	gCaL = .01;
	gNaP = 0.1;
	gCAN = 0.01;
	gSK = .4;
	gKCC2 = c_som * 1.0;
	gKCC2_dend = c_dend * 1.0;
	gHCN = 0.1;
	gTRPC3 = 0.1;
	gChR2 = 0.04;
	gArch = 0.04;
	E_leak = -60;
	g_L = 0.068;
	hcan = 1;
	OP1 = 0.00;
	OP2 = 0.00;
	CL1 = 0.99;
	CL2 = 0.01;
	Pchr2 = 0.1;
	Carch = 0.9;
	Oarch = 0.0;
	D_pre = D_0;
	F_pre = F_0;
	g_GABA_str = 0;
	g_GABA_gpe = 0;
}

// CHECK IF NEURON HAS SPIKED
int Neuron::spike(double vth, double dt, double t)
{
	double vpre = v;
	step(dt, t);
	return (vpre < vth && v >= vth);
}

// SET INITIAL CONDITIONS OF NEURON
void Neuron::init(double gca, double gcal, double glk, double gnap, double gcan, double gchr2, double garch, double gsk, double ghcn, double gtrpc3, double gkcc2, double gkcc2_dend)
{
	gCa = gca;
	gCaL = gcal;
	g_L = glk;
	gNaP = gnap;
	gCAN = gcan;
	gSK = gsk;
	gHCN = ghcn;
	gTRPC3 = gtrpc3;
	gChR2 = gchr2;
	gArch = garch;
	gKCC2 = c_som * gkcc2;			  // added to take in kcc2 changes
	gKCC2_dend = c_dend * gkcc2_dend; // added to take in kcc2 changes
	GCLtonic = 0;
	GCLtonic_dend = 0;
	GCLtonicSOMA = 0; // added for GCL tonic soma channel
	m = rnd();
	h = rnd();
	km = rnd();
	kh = rnd();
	s = rnd();
	napm = rnd();
	naph = rnd();
	mc = rnd();
	mSK = rnd();
	mcan = rnd();
	h_can = 0.05;
	hc = rnd();
	n = rnd();
	Cain = 5e-5 * rnd();
	CLin = 120; // 15.8e-3;
	OP1 = 0.00;
	OP2 = 0.00;
	CL1 = 0.99;
	CL2 = 0.01;
	Pchr2 = 0.1;
	Carch = 0.9;
	Oarch = 0.0;
	D_pre = D_0;
	F_pre = F_0;
}

// EULER STEP NEURONS
void Neuron::step(double dt, double t)
{
	// ALLOW FOR NOISE IN CERTAIN PARAMETERS
	std::normal_distribution<double> wiener_process(0, sqrt(dt));

	T += dt;

	// Reversals updated at each time step
	ECa = 26.54 * log(Caout / Cain) / 2;
	ECL = -26.54 * log(CLout / CLin);
	ECL_SOMdend = -26.54 * log(CLin_dend / CLin);
	ECL_dendSOM = -26.54 * log(CLin / CLin_dend);
	ECL_dend = -26.54 * log(CLout / CLin_dend);
	EGABA = -26.54 * log((pcl * CLout + phco3 * hco3out) / (pcl * CLin + phco3 * hco3in));
	EGABA_dend = -26.54 * log((pcl * CLout + phco3 * hco3out) / (pcl * CLin_dend + phco3 * hco3in));

	// Some channels defined as separate variables
	INaF = gNaF * m * m * m * h * s * (v - ENa);
	INaP = gNaP * napm * napm * napm * naph * (v - ENa);
	ICa = gCa * mc * hc * (v - ECa);
	ITRPC3 = 0.0; // gTRPC3*(v-ETRPC3)*TRPC3_BLOCK; No trpc3 in soma in this version of the model
	ITRPC3_dend = gTRPC3 * (v_dend - ETRPC3);
	IKdr = gKdr * km * km * km * km * kh * (v - EK);
	ISK = (gSK / (1. + pow(K_SK / Cain, nc_SK))) * (v - EK);
	Ih = gHCN * m_hcn * (v - EHCN);
	IL = g_L * (v - E_leak);

	// double Iexc = gSTN * (v - Eexc);
	//  Som and Dendrite communcitation
	// ISOMdend = ((gSD / c_som) / ((1 - CsCd))) * (v_dend - v);
	// IDENDsom = ((gSD / c_som) / ((CsCd))) * (v - v_dend);

	// gSD = 0;
	ISOMdend = gSD * (v_dend - v);
	IDENDsom = gSD * (v - v_dend);

	// Cl som and dendrite communication
	I_CL_SOMdend = gSD / (CsCd) * ((v_dend - v) + ECL_SOMdend);		// Cl current from som to dend
	I_CL_dendSOM = gSD / (1 + CsCd) * ((v - v_dend) + ECL_dendSOM); // Cl current from dend to som

	// Noise updates (in-vivo)
	soma_current_noise += soma_current_intensity_theta * (soma_current_noise_mu - soma_current_noise) * dt + soma_current_intensity * wiener_process(dist_gen);
	dend_current_noise += dend_current_intensity_theta * (dend_current_noise_mu - dend_current_noise) * dt + dend_current_intensity * wiener_process(dist_gen);
	g_tonic_stn += gstn_theta * (gstn_mu - g_tonic_stn) * dt + gstn_sigma * wiener_process(dist_gen);
	GCLtonicSOMA += gcl_theta * (gcl_mu - GCLtonicSOMA) * dt + gcl_sigma * wiener_process(dist_gen);
	GCLtonic_dend += gcl_theta_dend * (gcl_mu_dend - GCLtonic_dend) * dt + gcl_sigma_dend * wiener_process(dist_gen);
	GCLtonic = 1.0 * GCLtonicSOMA;
	//}

	// Update dynamical variables
	m += (x_inf(v, mV12, mk) - m) * (1. - exp(-dt / tau_inf2(mtau0, v, mtau1, mphi, msigma0, msigma1)));
	h += (x_inf(v, hV12, hk) - h) * (1. - exp(-dt / tau_inf2(htau0, v, htau1, hphi, hsigma0, hsigma1)));
	s += (x_inf2(v, sV12, sk, smin) - s) * (1. - exp(-dt / tau_inf2(stau0, v, stau1, sphi, ssigma0, ssigma1)));

	// fast potassium activation/inactivation
	km += (x_inf(v, kmV12, kmk) - km) * (1. - exp(-dt / tau_inf2(kmtau0, v, kmtau1, kmphi, kmsigma0, kmsigma1)));
	kh += (x_inf2(v, khV12, khk, khmin) - kh) * (1. - exp(-dt / tau_inf2(khtau0, v, khtau1, khphi, khsigma0, khsigma1)));

	// INaP activation/inactivation
	napm += (x_inf(v, napmV12, napmk) - napm) * (1. - exp(-dt / tau_inf2(napmtau0, v, napmtau1, napmphi, napmsigma0, napmsigma1)));
	naph += (x_inf2(v, naphV12, naphk, naphmin) - naph) * (1. - exp(-dt / tau_inf2(naphtau0, v, naphtau1, naphphi, naphsigma0, naphsigma1)));

	// High voltage Ca activation & inactivation
	mc += (x_inf(v, mcV12, mck) - mc) * (1. - exp(-dt / mctau));
	hc += (x_inf(v, hcV12, hck) - hc) * (1. - exp(-dt / hctau));

	//	//I_HCN
	m_hcn += (x_inf(v, mhcn_V12, mhcn_k) - m_hcn) * (1. - exp(-dt / tau_inf2(mhcn_tau0, v, mhcn_tau1, mhcn_phi, mhcn_sigma0, mhcn_sigma1)));

	// gating variable for calcium activated potassium channel
	mSK += (1 / (1. + pow(K_SK / Cain, nc_SK)) - mSK) * (1. - exp(-dt / tauSK_inf));

	// CAN Channel Activation
	mcan = 1 / (1. + pow(K_CAN / Cain, nc));
	h_can += (x_inf(v, canh_V12, canh_k) - h_can) * (1. - exp(-dt / tau_inf3(can_tau0, v, can_tau1, can_tau_V12, can_tau_k)));

	// Ca2+
	// Calcium dynamics only in the soma (no ca dep. currents in dend)
	Cain += (-alphaCa * ICa + (Cain0 - Cain) / tauCa) * dt; // units are in mM or 10-3M

	// KCC2 pump and CL dynamics
	double x_som = (EHCO3 - EGABA) / (EHCO3 - ECL);
	double x_dend = (EHCO3 - EGABA_dend) / (EHCO3 - ECL_dend);
	CLin += (-alphaCL * (gKCC2 * (ECL - EK) - x_som * g_GABA_snr * (v - ECL) - x_som * g_GABA_gpe * (v - ECL) - x_som * GCLtonic * (v - ECL)) - (CLin - CLin_dend) / (100 * 2)) * dt;
	CLin_dend += (-alphaCL_dend * (gKCC2_dend * (ECL_dend - EK) - x_dend * g_GABA_snr_dend * (v_dend - ECL_dend) - x_dend * g_GABA_str * (v_dend - ECL_dend) - x_dend * GCLtonic_dend * (v - ECL_dend)) - (CLin_dend - CLin) / (40 * 2)) * dt;

	// depression and faciliation
	D_pre += ((D_0 - D_pre) / tau_D) * dt;

	// dend dynamics
	F_pre += ((F_0 - F_pre) / tau_F) * dt;

	// g_GABA_gpe *= exp(-dt / 8); // exp(-dt / tausyn);
	g_GABA_gpe *= exp(-dt / tausyn);
	g_GABA_str *= exp(-dt / tausyn_dend);
	gSTN *= exp(-dt / tauexc);
	g_GABA_snr *= exp(-dt / 3); // fixed
	g_GABA_snr_dend *= exp(-dt / 7.2);

	// excitatory train (STN->SNr)
	// gEXCDEND = 0;
	for (int i = 0; i < network_size; i++)
	{
		if (STN_f * dt / 1000 >= rnd())
		{
			gSTN += c_dend * STN_strength;
		}
	}

	// GPe Stimulation
	if (gpe_stim > 0)
	{
		if (gpe_poisson == 1)
		{

			if (t >= gpe_start_time + gpe_base_length + gpe_stim_count * gpe_trial_length)
			{
				GPe_ON = 1;
				GCLtonic = 0;
				GCLtonicSOMA = 0;
			}

			if (t > gpe_start_time + gpe_base_length + gpe_stim_count * gpe_trial_length + gpe_stim_length)
			{
				GPe_ON = 0;
				gpe_stim_count += 1;
			}

			if ((gpe_Freq * dt / 1000 >= 0 * rnd()) && (GPe_ON > 0))
			{
				// g_GABA_gpe += c_som * W_gpe * D_pre;
				g_GABA_gpe = c_som * W_gpe * D_pre;
				D_pre -= alpha_D * (D_pre - D_min) * dt;
				GCLtonic = 0;
				GCLtonicSOMA = 0;
			}
		}
		else
		{
			if ((t >= gpe_start_time + gpe_base_length + gpe_stim_count * gpe_trial_length + gpe_pulses * gpe_pulse_length) && (GPe_ON == 0))
			{
				GPe_ON = 1;
				GCLtonic = 0;
				GCLtonicSOMA = 0;
			}
			if (t >= gpe_start_time + gpe_base_length + gpe_stim_count * gpe_trial_length + gpe_pulses * gpe_pulse_length + gpe_width * dt)
			{
				GPe_ON = 0;
				gpe_pulses += 1;
				GCLtonic = 0;
				GCLtonicSOMA = 0;
			}
			if (gpe_pulses >= gpe_max_pulses)
			{
				GPe_ON = 0;
				gpe_pulses = 0;
			}

			if (t > gpe_start_time + gpe_stim_length + gpe_stim_count * gpe_trial_length + gpe_stim_length)
			{
				GPe_ON = 0;
				gpe_pulses = 0;
				gpe_stim_count += 1;
			}
			if (GPe_ON > 0)
			{
				g_GABA_gpe += c_som * W_gpe * D_pre;
				D_pre -= alpha_D * (D_pre - D_min) * dt;
				GCLtonic = 0;
				GCLtonicSOMA = 0;
				GPe_ON = 0;
			}
		}
	}

	if (str_stim > 0)
	{

		if (str_poisson == 1)
		{

			if (t >= str_start_time + str_base_length + str_stim_count * str_trial_length)
			{
				STR_ON = 1;
				GCLtonic_dend = 0;
			}

			if (t > str_start_time + str_base_length + str_stim_count * str_trial_length + str_stim_length)
			{
				STR_ON = 0;
				str_stim_count += 1;
			}

			if ((str_Freq * dt / 1000 >= 0 * rnd()) && (STR_ON > 0))
			{
				// g_GABA_str += c_dend * W_str * F_pre;
				g_GABA_str = c_dend * W_str * F_pre;
				F_pre += alpha_F * (1 - F_pre) * dt;
				GCLtonic_dend = 0;
			}
		}
		else
		{
			if ((t >= str_start_time + str_base_length + str_stim_count * str_trial_length + str_pulses * str_pulse_length) && (GPe_ON == 0))
			{
				STR_ON = 1;
				GCLtonic_dend = 0;
			}
			if (t >= str_start_time + str_base_length + str_stim_count * str_trial_length + str_pulses * str_pulse_length + str_width * dt)
			{
				STR_ON = 0;
				str_pulses += 1;
				GCLtonic_dend = 0;
			}
			if (str_pulses >= str_max_pulses)
			{
				STR_ON = 0;
				str_pulses = 0;
			}

			if (t > str_start_time + str_stim_length + str_stim_count * str_trial_length + str_stim_length)
			{
				STR_ON = 0;
				str_pulses = 0;
				str_stim_count += 1;
			}
			if (STR_ON > 0)
			{
				g_GABA_str += c_dend * W_str * F_pre;
				F_pre += alpha_F * (1 - F_pre) * dt;
				GCLtonic_dend = 0;
				STR_ON = 0;
			}
		}
	}

	// Update dv/dt and perform an Euler step
	dv -= INaF; // * (v - ENa);
	dv -= INaP; //* (v - ENa);
	dv -= IKdr; //* (v - EK);
	dv -= ICa;
	dv -= ISK; //* (v - EK);
	dv -= IL;  //* (v - E_leak);
	dv -= IDENDsom;
	dv -= g_GABA_gpe * (v - EGABA);
	dv -= g_GABA_snr * (v - EGABA);
	dv -= GCLtonicSOMA * (v - EGABA);
	dv -= Ih;
	dv -= ITRPC3;
	dv += IAPP;
	dv = dv / c_som;
	v += dv * dt;
	v += soma_current_noise / c_som; // soma_current_intensity * wiener_process(dist_gen) / c_som;

	// Update dv_dend/dt and perform and Euler step
	dv_dend -= g_GABA_snr_dend * (v_dend - EGABA_dend);
	dv_dend -= g_GABA_str * (v_dend - EGABA_dend);
	dv_dend -= GCLtonic_dend * (v_dend - EGABA_dend);
	dv_dend -= ISOMdend;
	dv_dend -= ITRPC3_dend;
	dv_dend -= gSTN * (v_dend - Eexc); // STN connections
	dv_dend -= g_tonic_stn * (v_dend - Eexc);
	dv_dend += IAPP_DEND;
	dv_dend = dv_dend / c_dend;
	v_dend += dv_dend * dt;
	v_dend += dend_current_noise / c_dend; // dend_current_intensity * wiener_process(dist_gen) / c_dend;
	t_lastsp += dt;
}

int main(int argc, char **argv)
{
	double dt = .05, DT = 1, T = 10000, status = 1, dynamics = 1;
	int size = 100, seed = 0;

	std::string directory = "test";
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-DT") == 0)
			DT = atof(argv[++i]);
		else if (strcmp(argv[i], "-dt") == 0)
			dt = atof(argv[++i]);
		else if (strcmp(argv[i], "-s") == 0)
			size = atoi(argv[++i]);
		else if (strcmp(argv[i], "-T") == 0)
			T = atof(argv[++i]) * 1000;
		else if (strcmp(argv[i], "-sd") == 0)
			seed = atoi(argv[++i]);
		else if (strcmp(argv[i], "-status") == 0)
		{
			status = atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-dynamics") == 0)
		{
			dynamics = atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-save_dir") == 0)
		{
			directory = argv[++i];
		}
	}

	if (seed != 0)
	{
		srand(seed);
	}

	int freq = int(DT / dt);

	Population pop(size, directory);

	std::ofstream cell_dynamics[size];
	std::ofstream spike_times[size];
	std::ofstream meta_data[size];

	for (int i = 0; i < size; i++)
	{
		std::string save_dir = directory + "/Neuron_" + std::to_string(i);
		cell_dynamics[i].open(save_dir + "/cell_dynamics.txt");
		spike_times[i].open(save_dir + "/spike_times.txt");
		meta_data[i].open(save_dir + "/meta_data.txt");
	}

	int sp = 0;
	int h[size];
	for (int i = 0; i < size; i++)
	{
		h[i] = 0;
		meta_data[i] << "gTON_STN_MEAN_nS_pF:\t" << pop.net[i].gstn_mu / c_dend << std::endl;
		meta_data[i] << "gTON_STN_SIGMA:\t" << pop.net[i].gstn_sigma << std::endl;
		meta_data[i] << "gTON_STN_THETA:\t" << pop.net[i].gstn_theta << std::endl;

		meta_data[i] << "gTON_CL_S_MEAN_nS_pF:\t" << pop.net[i].gcl_mu / c_som << std::endl;
		meta_data[i] << "gTON_CL_S_SIGMA:\t" << pop.net[i].gcl_sigma << std::endl;
		meta_data[i] << "gTON_CL_S_THETA:\t" << pop.net[i].gcl_theta << std::endl;

		meta_data[i] << "gTON_CL_D_MEAN_nS_pF:\t" << pop.net[i].gcl_mu_dend / c_dend << std::endl;
		meta_data[i] << "gTON_CL_D_SIGMA:\t" << pop.net[i].gcl_sigma_dend << std::endl;
		meta_data[i] << "gTON_CL_D_THETA:\t" << pop.net[i].gcl_theta_dend << std::endl;

		meta_data[i] << "gKCC2_S_nS_pF:\t" << pop.net[i].gKCC2 / c_som << std::endl;
		meta_data[i] << "gKCC2_D_nS_pF:\t" << pop.net[i].gKCC2_dend / c_dend << std::endl;
		meta_data[i] << "gTRPC3_nS_pF:\t" << pop.net[i].gTRPC3 / c_dend << std::endl;
		meta_data[i] << "gHCN_nS_pF:\t" << pop.net[i].gHCN / c_som << std::endl;
		meta_data[i] << "gCA_nS_pF:\t" << pop.net[i].gCa / c_som << std::endl;
		meta_data[i] << "gL_nS_pF:\t" << pop.net[i].g_L / c_som << std::endl;
		meta_data[i] << "gSK_nS_pF:\t" << pop.net[i].gSK / c_som << std::endl;
		meta_data[i] << "gNAP_nS_pF:\t" << pop.net[i].gNaP / c_som << std::endl;
		meta_data[i] << "gNAF_nS_pF:\t" << pop.net[i].gNaF / c_som << std::endl;
		meta_data[i] << "gKDR_nS_pF:\t" << pop.net[i].gKdr / c_som << std::endl;
		meta_data[i] << "gSD_nS:\t" << pop.net[i].gSD << std::endl;

		meta_data[i] << "soma_noise_intensity:\t" << pop.net[i].soma_current_intensity << std::endl;
		meta_data[i] << "soma_noise_intensity_theta:\t" << pop.net[i].soma_current_intensity_theta << std::endl;
		meta_data[i] << "dend_noise_intensity:\t" << pop.net[i].dend_current_intensity << std::endl;
		meta_data[i] << "dend_noise_intensity_theta:\t" << pop.net[i].dend_current_intensity_theta << std::endl;

		meta_data[i] << "Eleak_mV:\t" << pop.net[i].E_leak << std::endl;
		meta_data[i] << "CL_in_S:\t" << pop.net[i].CLin << std::endl;
		meta_data[i] << "CL_in_D:\t" << pop.net[i].CLin_dend << std::endl;
		meta_data[i] << "Iapp:\t" << pop.net[i].IAPP << std::endl;
		meta_data[i] << "Iapp_dend:\t" << pop.net[i].IAPP_DEND << std::endl;

		meta_data[i] << "gpe_stim:\t" << pop.net[i].gpe_stim << std::endl;
		meta_data[i] << "gpe_stim_freqs:\t" << pop.net[i].gpe_Freq << std::endl;
		meta_data[i] << "gpe_poisson:\t" << pop.net[i].gpe_poisson << std::endl;
		meta_data[i] << "W_gpe:\t" << pop.net[i].W_gpe << std::endl;
		meta_data[i] << "gpe_start_time:\t" << pop.net[i].gpe_start_time << std::endl;
		meta_data[i] << "gpe_base_length:\t" << pop.net[i].gpe_base_length << std::endl;
		meta_data[i] << "gpe_stim_length:\t" << pop.net[i].gpe_stim_length << std::endl;
		meta_data[i] << "gpe_post_length:\t" << pop.net[i].gpe_post_length << std::endl;

		meta_data[i] << "str_stim:\t" << pop.net[i].str_stim << std::endl;
		meta_data[i] << "str_stim_freqs:\t" << pop.net[i].str_Freq << std::endl;
		meta_data[i] << "str_poisson:\t" << pop.net[i].str_poisson << std::endl;
		meta_data[i] << "W_str:\t" << pop.net[i].W_str << std::endl;
		meta_data[i] << "str_start_time:\t" << pop.net[i].str_start_time << std::endl;
		meta_data[i] << "str_base_length:\t" << pop.net[i].str_base_length << std::endl;
		meta_data[i] << "str_stim_length:\t" << pop.net[i].str_stim_length << std::endl;
		meta_data[i] << "str_post_length:\t" << pop.net[i].str_post_length << std::endl;
		meta_data[i] << "tausyn:\t" << pop.net[i].tausyn << std::endl;
		meta_data[i] << "tauexc:\t" << pop.net[i].tauexc << std::endl;
		meta_data[i] << "tausyn_dend:\t" << pop.net[i].tausyn_dend << std::endl;
		meta_data[i].close();
	}

	double time_step = 0;
	double tot_steps = (int)(T / dt);
	for (double t = 0; t <= T; t += dt)
	{
		time_step += 1;
		if (status > 0)
		{
			std::cout << "Completion:\t" << (time_step / tot_steps) * 100 << "%" << '\r' << std::flush;
		}

		for (int i = 0; i < size; i++)
		{
			h[i] = 0;
		}

		sp = 0;
		int spk[size];

		sp += pop.step(dt, spk, t);

		for (int i = 0; i < size; i++)
		{
			h[i] += spk[i];
			if (h[i] >= 1)
			{
				spike_times[i] << (t / 1000) << std::endl;
			}
		}

		if (int(t / dt) % (freq) == 0 && dynamics > 0)
		{
			for (int i = 0; i < size; i++)
			{
				cell_dynamics[i] << (t / 1000)
								 << "\t" << pop.net[i].v
								 << "\t" << pop.net[i].v_dend
								 << "\t" << pop.net[i].dv
								 << "\t" << pop.net[i].dv_dend
								 << "\t" << pop.net[i].g_GABA_snr / c_som
								 << "\t" << pop.net[i].g_GABA_gpe / c_som / pop.net[i].D_pre
								 << "\t" << pop.net[i].g_GABA_str / c_dend / pop.net[i].F_pre
								 << "\t" << pop.net[i].g_tonic_stn / c_dend
								 << "\t" << pop.net[i].GCLtonicSOMA / c_som
								 << "\t" << pop.net[i].GCLtonic_dend / c_dend
								 << "\t" << pop.net[i].EGABA
								 << "\t" << pop.net[i].EGABA_dend
								 << "\t" << pop.net[i].CLin
								 << "\t" << pop.net[i].CLin_dend
								 << "\t" << pop.net[i].D_pre
								 << "\t" << pop.net[i].F_pre
								 << "\t" << pop.net[i].soma_current_noise
								 << "\t" << pop.net[i].dend_current_noise
								 << std::endl;
			}
		}
	}

	for (int i = 0; i < pop.size; i++)
	{
		cell_dynamics[i].close();
		spike_times[i].close();
	}
}
