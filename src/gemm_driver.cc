#include "gemm_driver.h"
#include "gemm_config.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
extern void cblas_sgemm_opt(layout_t Layout, trans_t Trans_a, trans_t Trans_b,
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc);

#define INTEL_AVX2
static double get_peak_gflops_fp32(double freq_mhz){
#ifdef INTEL_AVX2
    return 2/*2 port*/*8/*fp32 for 256 bit*/*2/*fma*/*freq_mhz/1024.0;
#endif
}

class arg_parser{
#define ARG_VALUE_INIT "n/a"
public:
    struct arg_store{
        std::string arg_name;
        std::string value;
        std::string default_value;
        std::string help_str;
    };
    arg_parser(const char * _name):name(_name){};
    ~arg_parser(){}
    void insert_arg(const char * arg, const char * help, std::string default_value){
        arg_store a;
        a.arg_name = std::string("-") + arg;
        a.help_str = help;
        a.default_value = default_value;
        a.value = ARG_VALUE_INIT;
        arg_pair[a.arg_name] = a;
    }
    bool parse(int argc, char ** argv){
        for(int i=0;i<argc;i+=2){
            std::string arg_name = argv[i];
            if(arg_name == "--help" || arg_name == "-help"){
                usage();
                return false;
            }
            if(arg_pair.count(arg_name) == 0){
                std::cerr<<"unrecognized arg "<<arg_name<<std::endl;;
                usage();
                return false;
            }
            if((i+1) >= argc){
                std::cerr<<"no value specified for this arg"<<std::endl;
                usage();
                return false;
            }
            arg_pair[arg_name].value = argv[i+1];
        }
        return true;
    }
    std::string get_arg_str(const char * arg){
        std::string arg_name = std::string("-") + arg;
        if(arg_pair.count(arg_name) == 0){
            std::cerr<<"no such arg "<<arg_name<<std::endl;
            usage();
            assert(0 && "no such arg in parse arg");
        }
        std::string val = arg_pair[arg_name].value;
        if(val == ARG_VALUE_INIT)
            val = arg_pair[arg_name].default_value;
        return val; 
    }

    template<typename T>
    T get_arg(const char * arg){
        T value;
        std::stringstream ss(get_arg_str(arg));
        ss >> value;
        return value;
    }
    template<typename T>
    T get_arg_choice(const char * arg, std::unordered_map<std::string, T> c_map){
        std::string key = get_arg_str(arg);
        if(c_map.count(key) == 0){
            std::cerr<<"no such arg in choice map"<<key<<std::endl;
            usage();
            assert(0 && "no such arg in choice map");
        }
        T value = c_map[key];
        return value;
    }

    void usage(){
        std::cout<<name<<" args:"<<std::endl;
        std::cout<<"    --help, print usage and return"<<std::endl;
        for(auto & it : arg_pair){
            arg_store a = it.second;
            std::cout<<"    "<<it.first<<", "<<a.help_str<<
                " (default:"<<a.default_value<<")"
#if 0
                <<", cur:"<<a.value
#endif           
                <<std::endl;
        }
    }
    void dump_parsed(){
        std::cout<<"using args: "<<name<<" ";
        for(auto & it : arg_pair){
            arg_store a = it.second;
            std::cout<<" "<<it.first<<" "<<(a.value==ARG_VALUE_INIT?a.default_value:a.value);
        }
        std::cout<<std::endl;
    }
private:
    std::string name;
    std::unordered_map<std::string, arg_store> arg_pair;
};

struct bench_result{
    int             loops;
    double          gflops;
    double          time_ms;    // cost for 1 loop
    double          perf;       // percentage
    matrix_fp32_t * c;
    bench_result(int loops_, double gflops_, double time_ms_, double perf_, matrix_fp32_t * c_):
        loops(loops_),gflops(gflops_),time_ms(time_ms_),perf(perf_),c(c_){}
    bench_result(bench_result && rhs){
        loops = rhs.loops;
        gflops = rhs.gflops;
        time_ms = rhs.time_ms;
        perf = rhs.perf;
        c = rhs.c;
        rhs.c = nullptr;
    }
    ~bench_result(){
        if(c)
            delete c;
    }
};

#define LOOPS 6
#define LOOP_WARMUP 1

class gemm_problem_t{
public:
    gemm_problem_t(int m_, int n_, int k_, float alpha_, float beta_,
        layout_t layout_, trans_t trans_a_, trans_t trans_b_, size_t align_,
        double freq_)
    {
        m = m_;
        n = n_;
        k = k_;
        alpha = alpha_;
        beta = beta_;
        layout = layout_;
        trans_a = trans_a_;
        trans_b = trans_b_;
        align = align_;
        freq = freq_;

        gemm_desc = new gemm_desc_t(m,n,k,layout,trans_a,trans_b);
        int row, col, inc_row, inc_col;
        std::tie(row, col, inc_row, inc_col) = gemm_desc->get_a();
        A = new matrix_fp32_t(row, col, inc_row, inc_col, layout, trans_a, align);
        

        std::tie(row, col, inc_row, inc_col) = gemm_desc->get_b();
        B = new matrix_fp32_t(row, col, inc_row, inc_col, layout, trans_b, align);

        std::tie(row, col, inc_row, inc_col) = gemm_desc->get_c();
        C = new matrix_fp32_t(row, col, inc_row, inc_col, layout, TRANS_NO_TRANS, align);
    }
    ~gemm_problem_t(){
        delete A;
        delete B;
        delete C;
        delete gemm_desc;
    }

    template<typename F>
    bench_result run_bench(F gemm_func, bool validate_only = false){
        matrix_fp32_t * c_out = new matrix_fp32_t(*C);
        //std::cout<<"[blas]layout:"<<layout<<", trans_a:"<<trans_a<<", trans_b:"<<trans_b<<
        //    ", m:"<<m<<", n:"<<n<<", k:"<<k<<", alpha:"<<alpha<<", beta:"<<beta<<
        //    ", lda:"<<A->h_stride<<", ldb:"<<B->h_stride<<", ldc:"<<c_out->h_stride<<std::endl;
        if(validate_only){
            // validate mode, only care about result
            gemm_func(layout,trans_a,trans_b,
                m,n,k,
                alpha,
                A->data,A->h_stride,
                B->data,B->h_stride,
                beta,
                c_out->data, c_out->h_stride);
            return std::move(bench_result(0,0,0,0,c_out));
        }

        int i;
        for(i=0;i<LOOP_WARMUP;i++){
            gemm_func(layout,trans_a,trans_b,
                m,n,k,
                alpha,
                A->data,A->h_stride,
                B->data,B->h_stride,
                beta,
                c_out->data, c_out->h_stride);
        }
        double start_time = current_sec();
        for(i=0;i<LOOPS;i++){
            gemm_func(layout,trans_a,trans_b,
                m,n,k,
                alpha,
                A->data,A->h_stride,
                B->data,B->h_stride,
                beta,
                c_out->data, c_out->h_stride);
        }
        double cost_per_loop = (current_sec()-start_time) / LOOPS;
        unsigned long long flop = sgemm_flop(m,n,k,alpha,beta);
        double gflops = (double)flop/(cost_per_loop *1e9);
        double gflops_theory = get_peak_gflops_fp32(freq);
        delete c_out;
        return std::move(bench_result(LOOPS, gflops, cost_per_loop*1e3, gflops/gflops_theory*100, nullptr));
    }

private:
    matrix_fp32_t *A;   // M*N
    matrix_fp32_t *B;   // N*K
    matrix_fp32_t *C;   // M*N

    gemm_desc_t * gemm_desc;
    int m;
    int n;
    int k;
    float alpha;
    float beta;
    layout_t layout;
    trans_t trans_a;
    trans_t trans_b;
    size_t align;
    double freq;    // in MHz
};

// specilization for openblas cblas_sgemm function
typedef decltype(&cblas_sgemm) cblas_sgemm_t;
template<>
bench_result gemm_problem_t::run_bench<cblas_sgemm_t>(cblas_sgemm_t cblas_gemm_func, bool validate_only){
    auto gemm_func = [&](layout_t _layout, trans_t _trans_a, trans_t _trans_b,
        int _m, int _n, int _k,
        const float _alpha,
        const float * _A, int _lda,
        const float * _B, int _ldb,
        const float _beta,
        float * _C, int _ldc) -> void
    {
        cblas_gemm_func(to_blas_layout(_layout),to_blas_transpose(_trans_a),to_blas_transpose(_trans_b),
            _m,_n,_k,_alpha,_A,_lda,_B,_ldb,_beta,_C,_ldc);
    };
    return run_bench(gemm_func, validate_only);
}

class gemm_bench{
public:
    struct config{
        int m=0;
        int n=0;
        int k=0;
        float alpha=.0f;
        float beta=.0f;
        layout_t layout=LAYOUT_ROW_MAJOR;
        trans_t trans_a=TRANS_NO_TRANS;
        trans_t trans_b=TRANS_NO_TRANS;
        // TODO: layout, trans
    };

    bool next_config(config * cfg){
        static int ITER_START = 32;
        static int ITER_STEP = 32;
        static int ITER_END = 4096;
        static float ALPHA = 1.0f;
        static float BETA  = 1.0f;

        static int current_iter = ITER_START;
        static bool need_stop = false;
        if(need_stop)
            return false;

        cfg->m = current_iter;
        cfg->n = current_iter;
        cfg->k = current_iter;
        cfg->alpha = ALPHA;
        cfg->beta = BETA;
        cfg->layout = LAYOUT_ROW_MAJOR;
        cfg->trans_a = TRANS_NO_TRANS;
        cfg->trans_b = TRANS_NO_TRANS;

        auto update_func = [&](){
            int step;
            if(current_iter < 256)
                step = ITER_STEP;
            else if(current_iter < 512)
                step = ITER_STEP * 2;
            else if(current_iter < 1024)
                step = ITER_STEP * 4;
            else if(current_iter < 2048)
                step = ITER_STEP * 4;
            else
                step = ITER_STEP * 8;

            current_iter += step;
            if(current_iter > ITER_END)
                need_stop = true;
        };
        update_func();

        return true;
    }
    bool next_config_valid(config * cfg){
        int Ms[] = {64,128,256,512,768,1024};
        int Ns[] = {64,128,256,512,768};
        int Ks[] = {64,128,256,512,768};
        float alphas[] = {1.0f, 2.1f};
        float betas[] = {1.0f, .0f, 1.6f};
#define ARRAY_LEN(arr) (sizeof(arr)/sizeof(arr[0]))
        static size_t  M_idx=0;
        static size_t  N_idx=0;
        static size_t  K_idx=0;
        static size_t  alpha_idx=0;
        static size_t  beta_idx=0;
        static bool need_stop = false;
        if(need_stop)
            return false;

        cfg->m = Ms[M_idx];
        cfg->n = Ns[N_idx];
        cfg->k = Ks[K_idx];
        cfg->alpha = alphas[alpha_idx];
        cfg->beta = betas[beta_idx];
        cfg->layout = LAYOUT_ROW_MAJOR;
        cfg->trans_a = TRANS_NO_TRANS;
        cfg->trans_b = TRANS_NO_TRANS;

        // next
        if(++beta_idx >= ARRAY_LEN(betas)){
            beta_idx = 0;
            if(++alpha_idx >= ARRAY_LEN(alphas)){
                alpha_idx = 0;
                if(++K_idx >= ARRAY_LEN(Ks)){
                    K_idx = 0;
                    if(++N_idx >= ARRAY_LEN(Ns)){
                        N_idx = 0;
                        if(++M_idx >= ARRAY_LEN(Ms)){
                            M_idx = 0;
                            need_stop = true;
                        }
                    }
                }
            }
        }
#undef ARRAY_LEN

        return true;
    }

    // note: this is preferd size in use of goto algo. small than this may still run
    inline int req_l1(){
        return BLOCK_K*NR*sizeof(float);
    }
    inline int req_l2(){
        return BLOCK_K*BLOCK_M*sizeof(float);
    }
    inline int req_l3(){
        return BLOCK_K*BLOCK_N*sizeof(float);
    }
    void run(double freq, bool validate_only = false){
        config cfg;
        printf("MC:%d, NC:%d, KC:%d, MR:%d, NR:%d\n", BLOCK_M, BLOCK_N, BLOCK_K, MR, NR);
        printf("require: L1:%.1fKB(KC*NR*4), L2:%.1fKB(KC*MC*4), L3:%.1fKB(KC*NC*4)\n", req_l1()/1024.0, req_l2()/1024.0, req_l3()/1024.0);
        printf("    M    N    K alpha beta   gflops(%%)   gflops_ref(%%)\n");
        while(1){
            bool have_next = validate_only?
                next_config_valid(&cfg):
                next_config(&cfg);
            if(!have_next)
                break;
            gemm_problem_t gemm_prob(cfg.m,cfg.n,cfg.k,cfg.alpha,cfg.beta,cfg.layout,cfg.trans_a,cfg.trans_b,32,freq);
            bench_result rtn_ref = gemm_prob.run_bench(cblas_sgemm, validate_only);
            bench_result rtn_opt = gemm_prob.run_bench(cblas_sgemm_opt, validate_only);
            
            printf(" %4d %4d %4d  %.1f  %.1f %6.2f(%2.4f) %6.2f(%2.4f)",
                cfg.m,cfg.n,cfg.k,cfg.alpha,cfg.beta,
                rtn_opt.gflops,rtn_opt.perf,rtn_ref.gflops,rtn_ref.perf);
            if(validate_only){
                bool result = valid_matrix(rtn_ref.c, rtn_opt.c, 0.001f);
                if(result)
                    printf("  <valid>");
                else
                    printf("  <fail>");
            }
            printf("\n");
        }
    }

};

#define MEM_ALIGN_BYTE "32"
int main(int argc, char ** argv){
    arg_parser args("gemm");
    args.insert_arg("m", "M value of gemm, int", "512");
    args.insert_arg("n", "N value of gemm, int", "512");
    args.insert_arg("k", "K value of gemm, int", "512");
    args.insert_arg("a", "ALPHA value of gemm, float", "1.0");
    args.insert_arg("b", "BETA value of gemm, float", "0");
    args.insert_arg("f", "CPU frequency, in MHz", "2600");

    args.insert_arg("layout", "layout, row|col", "row");
    args.insert_arg("ta", "translation for A, no|trans", "no");
    args.insert_arg("tb", "translation for B, no|trans", "no");
    args.insert_arg("align", "memory alignment for matrix, in byte", MEM_ALIGN_BYTE);
    args.insert_arg("valid", "validate the result", "0");
    args.insert_arg("bench", "benchmark mode, for all config", "1");

    if(!args.parse(argc-1, argv+1)) return -1;
    //args.dump_parsed();

    int m = args.get_arg<int>("m");
    int n = args.get_arg<int>("n");
    int k = args.get_arg<int>("k");
    float alpha = args.get_arg<float>("a");
    float beta  = args.get_arg<float>("b");
    double freq = args.get_arg<double>("f");

    layout_t layout = args.get_arg_choice<layout_t>("layout", {
                        {"row", LAYOUT_ROW_MAJOR},
                        {"col", LAYOUT_COL_MAJOR}
                        });
    std::unordered_map<std::string, trans_t> trans_map = {{"no", TRANS_NO_TRANS},{"trans", TRANS_TRANS}};
    trans_t trans_a = args.get_arg_choice<trans_t>("ta", trans_map);
    trans_t trans_b = args.get_arg_choice<trans_t>("tb", trans_map);
    int align = args.get_arg<int>("align");
    bool valid = (args.get_arg<int>("valid")==1) ? true:false;
    bool is_bench = (args.get_arg<int>("bench")==1) ? true:false;

    // force single thread openblas
    openblas_set_num_threads(1);

    if(is_bench){
        gemm_bench gb;
        gb.run(freq, valid);
        return 0;
    }
    args.dump_parsed();

    gemm_problem_t gemm_prob(m,n,k,alpha,beta,layout,trans_a,trans_b,align,freq);
    bench_result rtn_ref = gemm_prob.run_bench(cblas_sgemm, valid);
    bench_result rtn_opt = gemm_prob.run_bench(cblas_sgemm_opt, valid);

    if(valid){
        bool result = valid_matrix(rtn_ref.c, rtn_opt.c, 0.001f);
        std::cout<<"valid result:"<<(result?"ok":"diff")<<std::endl;
    }else{
        std::cout<<"[ref] gflops:"<<rtn_ref.gflops<<", time:"<<rtn_ref.time_ms<<"ms"<<std::endl;
        std::cout<<"[opt] gflops:"<<rtn_opt.gflops<<", time:"<<rtn_opt.time_ms<<"ms"<<std::endl;
    }

    return 0;
}
