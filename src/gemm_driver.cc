#include "gemm_driver.h"
#include "gemm_config.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <unistd.h>
#include <string>
#include <sstream>
#include <unordered_map>
#include <functional>

template<typename T>
bool valid_matrix(const matrix_t<T> * lhs, const matrix_t<T> * rhs, double delta){
    int i;
    int errs = 0;
    assert(lhs && rhs);
    size_t elements = matrix_elem_t()(lhs->row, lhs->col, lhs->ldim, lhs->layout, lhs->trans);
    assert(elements == matrix_elem_t()(rhs->row, rhs->col, rhs->ldim, rhs->layout, rhs->trans));
    for(i=0;i<(int) (elements);i++){
        double d = double(lhs->data[i] - rhs->data[i]);
        d = ABS(d);
        if(d>delta){
            if(errs<10)
                std::cout<<"["<<i<<"] result diff, left:"<<lhs->data[i]<<", right:"<<rhs->data[i]<<", delta:"<<d<<std::endl;
            errs++;
        }
    }
    return errs==0;
}

// the last parameter is added only for convenience
extern void cblas_sgemm_opt(layout_t Layout, trans_t Trans_a, trans_t Trans_b,
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx);

typedef decltype(&cblas_sgemm) cblas_sgemm_t;
//typedef decltype(&cblas_sgemm_opt) cblas_sgemm_opt_t;
typedef std::function<void(layout_t Layout, trans_t Trans_a, trans_t Trans_b,
                int M, int N, int K,
                float alpha,
                const float *A, int lda,
                const float *B, int ldb,
                float beta,
                float *C, int ldc,
                const gemm_context_t * ctx)
            > cblas_sgemm_opt_t;
// use std::function instead of func pointer type, can let lambda capture work

#define INTEL_AVX2

template<typename T>
class peak_gflops_t{
public:
    typedef T dtype;
    double operator() (double freq_mhz){
        return 0;
    }
};

template<>
class peak_gflops_t<float>{
public:
    double operator() (double freq_mhz){
#ifdef INTEL_AVX2
    return 2/*2 port*/*8/*fp32 for 256 bit*/*2/*fma*/*freq_mhz/1024.0;
#endif
    }
};


static inline std::string _to_arg_name(const char *arg){
    std::string s = std::string("-") + arg;
    return s;
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
        a.arg_name = _to_arg_name(arg);
        a.help_str = help;
        a.default_value = default_value;
        a.value = ARG_VALUE_INIT;
        arg_pair[a.arg_name] = a;
    }
    bool parse(int argc, char ** argv){
        this->parsed = true;
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
    // call 
    bool used_arg(const char * arg){
        if(!this->parsed){
            std::cerr<<"must call used_arg() after call parse()"<<std::endl;
            assert(0 && "should not happen\n");
        }
        std::string arg_name = _to_arg_name(arg);
        if(arg_pair.count(arg_name) == 0){
            std::cerr<<"no such arg "<<arg_name<<std::endl;
            usage();
            assert(0 && "no such arg in parse arg");
        }
        std::string val = arg_pair[arg_name].value;
        if(val == ARG_VALUE_INIT)
            return false;
        return true;
    }
    std::string get_arg_str(const char * arg){
        std::string arg_name = _to_arg_name(arg);
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
    bool parsed {false};
};

template<typename T>
struct bench_result{
    int             loops;
    double          gflops;
    double          time_ms;    // cost for 1 loop
    double          perf;       // percentage
    matrix_t<T>   * c {nullptr};
    bench_result(){}
    bench_result(int loops_, double gflops_, double time_ms_, double perf_, matrix_t<T> * c_):
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

#define LOOPS 5
#define LOOP_WARMUP 3

template <typename T>
class gemm_problem_t{
public:
    gemm_problem_t(gemm_context_t * ctx_)
    {
        this->ctx = ctx_;

        A = new matrix_t<T>(ctx->m, ctx->k, ctx->lda, ctx->layout, ctx->trans_a, ctx->alignment);
        B = new matrix_t<T>(ctx->k, ctx->n, ctx->ldb, ctx->layout, ctx->trans_b, ctx->alignment);
        C = new matrix_t<T>(ctx->m, ctx->n, ctx->ldc, ctx->layout, TRANS_NO_TRANS, ctx->alignment);

        loops = LOOPS;
        loop_warmup = LOOP_WARMUP;
    }
    ~gemm_problem_t(){
        delete A;
        delete B;
        delete C;
    }

    bench_result<T> run_single_case(cblas_sgemm_opt_t gemm_func, bool validate_only){
        matrix_t<T> * c_out = new matrix_t<T>(*C);
        //std::cout<<"[blas]layout:"<<layout<<", trans_a:"<<trans_a<<", trans_b:"<<trans_b<<
        //    ", m:"<<m<<", n:"<<n<<", k:"<<k<<", alpha:"<<alpha<<", beta:"<<beta<<
        //    ", lda:"<<A->h_stride<<", ldb:"<<B->h_stride<<", ldc:"<<c_out->h_stride<<std::endl;
        if(validate_only){
            // validate mode, only care about result
            gemm_func(ctx->layout,ctx->trans_a,ctx->trans_b,
                ctx->m,ctx->n,ctx->k,
                ctx->alpha,
                A->data,ctx->lda,
                B->data,ctx->ldb,
                ctx->beta,
                c_out->data, ctx->ldc, ctx);
            //https://en.cppreference.com/w/cpp/language/copy_elision 
            //return std::move(bench_result(0,0,0,0,c_out));
            return bench_result<T>(0,0,0,0,c_out);
        }

        int i;
        for(i=0;i<this->loop_warmup;i++){
            gemm_func(ctx->layout,ctx->trans_a,ctx->trans_b,
                ctx->m,ctx->n,ctx->k,
                ctx->alpha,
                A->data,ctx->lda,
                B->data,ctx->ldb,
                ctx->beta,
                c_out->data, ctx->ldc, ctx);
        }
        double start_time = current_sec();
        for(i=0;i<this->loops;i++){
            gemm_func(ctx->layout,ctx->trans_a,ctx->trans_b,
                ctx->m,ctx->n,ctx->k,
                ctx->alpha,
                A->data,ctx->lda,
                B->data,ctx->ldb,
                ctx->beta,
                c_out->data, ctx->ldc, ctx);
        }
        double cost_per_loop = (current_sec()-start_time) / this->loops;
        unsigned long long flop = sgemm_flop(ctx->m,ctx->n,ctx->k,ctx->alpha,ctx->beta);
        double gflops = (double)flop/(cost_per_loop *1e9);
        double gflops_theory = peak_gflops_t<T>()(ctx->frequency);
        delete c_out;
        //return std::move(bench_result(LOOPS, gflops, cost_per_loop*1e3, gflops/gflops_theory*100, nullptr));
        return bench_result<T>(this->loops, gflops, cost_per_loop*1e3, gflops/gflops_theory*100, nullptr);
    }
    // used for cblas api call
    bench_result<T> run_single_case(cblas_sgemm_t cblas_gemm_func, bool validate_only){
        auto gemm_func_wrapper = [&](layout_t _layout, trans_t _trans_a, trans_t _trans_b,
            int _m, int _n, int _k,
            const float _alpha,
            const float * _A, int _lda,
            const float * _B, int _ldb,
            const float _beta,
            float * _C, int _ldc,
            const gemm_context_t * ctx) -> void
        {
            (void)ctx;
            cblas_gemm_func(to_blas_layout(_layout),to_blas_transpose(_trans_a),to_blas_transpose(_trans_b),
                _m,_n,_k,_alpha,_A,_lda,_B,_ldb,_beta,_C,_ldc);
        };
        return run_single_case(gemm_func_wrapper, validate_only);
    }


//private:
    matrix_t<T> *A;   // M*N
    matrix_t<T> *B;   // N*K
    matrix_t<T> *C;   // M*N

    gemm_context_t * ctx;   // not own this

    int loop_warmup;
    int loops;
};


template<typename T>
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

        int lda=0;
        int ldb=0;
        int ldc=0;
        // TODO: layout, trans
    };

    bool next_config(config * cfg){
        static int ITER_START = 48;
        static int ITER_STEP = 48;
        static int ITER_END = 6144;
        static float ALPHA = 1.0f;
        static float BETA  = 1.0f;

        static int current_iter = ITER_START;
        static bool need_stop = false;
        if(need_stop)
            return false;

        cfg->m = current_iter;
        cfg->n = current_iter;
        cfg->k = current_iter;
        cfg->lda = current_iter;
        cfg->ldb = current_iter;
        cfg->ldc = current_iter;
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
#if 0
        int Ms[] = {64,128,256,512,768,1024};
        int Ns[] = {64,128,256,512,768};
        int Ks[] = {64,128,256,512,768};
#endif
        int Ms[] = {48,96,192,384,768};
        int Ns[] = {48,96,192,384,768};
        int Ks[] = {48,96,192,384,768};
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
        // TODO: lda, ldb, ldc based on row/col
        cfg->lda = cfg->k;
        cfg->ldb = cfg->n;
        cfg->ldc = cfg->n;

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
    //void run(std::vector<int> cpu_list, double freq, bool validate_only, bool no_ref, gemm_problem_t * single_problem = nullptr){
    void run(gemm_context_t *ctx, bool validate_only, bool no_ref, bool one_shot){
        auto cpu_list_to_str = [](const std::vector<int> & cpu_list_){
            std::string str;
            for(int i=0;i<cpu_list_.size();i++){
                str += std::to_string(cpu_list_[i]);
                if(i != (cpu_list_.size()-1) )
                    str += ",";
            }
            return str;
        };
        std::string cpu_list_str = cpu_list_to_str(ctx->cpu_list);
        printf("cpu:%s, freq: %.1fMHz, theoritical: %.3f gflops (avx256,fmadd)\n", cpu_list_str.c_str(), ctx->frequency, peak_gflops_t<T>()(ctx->frequency));
        printf("MC:%lu, NC:%lu, KC:%lu, MR:%lu, NR:%lu\n", ctx->mc, ctx->nc, ctx->kc, ctx->mr, ctx->nr);
        assert( ((ctx->mc % ctx->mr) == 0) && ((ctx->nc % ctx->nr) == 0) && "MC%%MR, NC%%NR must be zero\n");
        
        //printf("require: L1:%.1fKB(KC*NR*4), L2:%.1fKB(KC*MC*4), L3:%.1fKB(KC*NC*4)\n", req_l1()/1024.0, req_l2()/1024.0, req_l3()/1024.0);
        printf("    M    N    K alpha beta   gflops(%%)   gflops_ref(%%)\n");

        auto summary_func = [&](gemm_problem_t<T> * prob, bench_result<T> * r_ref, bench_result<T> * r_opt){
            printf(" %4lu %4lu %4lu  %.1f  %.1f %6.2f(%2.4f) %6.2f(%2.4f)",
                prob->ctx->m,prob->ctx->n,prob->ctx->k,prob->ctx->alpha,prob->ctx->beta,
                r_opt->gflops,r_opt->perf,r_ref?(r_ref->gflops):0,r_ref?(r_ref->perf):0);
            if(validate_only && r_ref){
                bool result = valid_matrix(r_ref->c, r_opt->c, 0.001f);
                if(result)
                    printf("  <valid>");
                else
                    printf("  <fail>");
            }
            printf("\n");
        };
        auto bench_single_func = [&](gemm_problem_t<T> * prob){
            if(no_ref){
                bench_result<T> rtn_opt = prob->run_single_case(cblas_sgemm_opt, validate_only);
                summary_func(prob, nullptr, &rtn_opt);
            }
            else{
                bench_result<T> rtn_ref = prob->run_single_case(cblas_sgemm, validate_only);
                bench_result<T> rtn_opt = prob->run_single_case(cblas_sgemm_opt, validate_only);
                summary_func(prob, &rtn_ref, &rtn_opt);
            }
        };
        while(1){
            if(one_shot){
                gemm_problem_t<T> gemm_prob(ctx);
                gemm_prob.loop_warmup *= 3;
                gemm_prob.loops  *= 6;
                bench_single_func(&gemm_prob);

                break;
            }
            else{
                config cfg;
#define VALID_BENCH_CONFIG
#ifdef VALID_BENCH_CONFIG
                bool have_next = next_config(&cfg);
#else
                bool have_next = validate_only?
                    next_config_valid(&cfg):
                    next_config(&cfg); 
#endif
                if(!have_next)
                    break;
                
                // change context here!!
                ctx->m = cfg.m;
                ctx->n = cfg.n;
                ctx->k = cfg.k;
                ctx->lda = cfg.lda;
                ctx->ldb = cfg.ldb;
                ctx->ldc = cfg.ldc;
                ctx->alpha = cfg.alpha;
                ctx->beta = cfg.beta;
                ctx->layout = cfg.layout;
                ctx->trans_a = cfg.trans_a;
                ctx->trans_b = cfg.trans_b;

                gemm_problem_t<T> gemm_prob(ctx);
                bench_single_func(&gemm_prob);
                //usleep(2000);
            }
        }
    }
};

#define MEM_ALIGN_BYTE 32
int main(int argc, char ** argv){
    arg_parser args("gemm");
    args.insert_arg("m", "M value of gemm, int", "512");
    args.insert_arg("n", "N value of gemm, int", "512");
    args.insert_arg("k", "K value of gemm, int", "512");
    args.insert_arg("lda", "leading dimension of a", "512");
    args.insert_arg("ldb", "leading dimension of b", "512");
    args.insert_arg("ldc", "leading dimension of c", "512");
    args.insert_arg("a", "ALPHA value of gemm, double", "1.0");
    args.insert_arg("b", "BETA value of gemm, double", "0");
    args.insert_arg("f", "CPU frequency, in MHz, double", "2600");

    args.insert_arg("layout", "layout, row|col", "row");
    args.insert_arg("ta", "translation for A, no|trans", "no");
    args.insert_arg("tb", "translation for B, no|trans", "no");
    args.insert_arg("align", "memory alignment for matrix, in byte", std::to_string(MEM_ALIGN_BYTE));
    args.insert_arg("valid", "validate the result", "0");
    args.insert_arg("no_ref", "do not run reference blas", "0");
    args.insert_arg("cpu", "run on which cpu", "2"); // TODO: cpu_list
    //args.insert_arg("bench", "benchmark mode, for all config", "1");
    args.insert_arg("mc", "MC", std::to_string(BLOCK_M));
    args.insert_arg("nc", "NC", std::to_string(BLOCK_N));
    args.insert_arg("kc", "KC", std::to_string(BLOCK_K));
    args.insert_arg("mr", "MR", std::to_string(MR));
    args.insert_arg("nr", "NR", std::to_string(NR));
    args.insert_arg("l1_size", "l1d cache size", std::to_string(L1_SIZE));
    args.insert_arg("l2_size", "l2 cache size", std::to_string(L2_SIZE));
    args.insert_arg("l3_size", "l3 cache size", std::to_string(L3_SIZE));
    args.insert_arg("cacheline_size", "cache line size", std::to_string(CACHELINE_SIZE));
    args.insert_arg("page_size","page size", std::to_string(PAGE_SIZE));
    args.insert_arg("tlb_entry_l1d","l1d tlb entry", std::to_string(L1D_TLB_ENTRY));

    if(!args.parse(argc-1, argv+1)) return -1;
    //args.dump_parsed();

    bool one_shot = args.used_arg("m")
                || args.used_arg("n") || args.used_arg("k")
                || args.used_arg("a") || args.used_arg("b") ;

    int m = args.get_arg<int>("m");
    int n = args.get_arg<int>("n");
    int k = args.get_arg<int>("k");
    int lda = args.get_arg<int>("lda");
    int ldb = args.get_arg<int>("ldb");
    int ldc = args.get_arg<int>("ldc");
    double alpha = args.get_arg<double>("a");
    double beta  = args.get_arg<double>("b");
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
    //bool is_bench = (args.get_arg<int>("bench")==1) ? true:false;
    bool no_ref = (args.get_arg<int>("no_ref") == 1) ? true:false;
    int cpu = args.get_arg<int>("cpu");
    int mc = args.get_arg<int>("mc");
    int nc = args.get_arg<int>("nc");
    int kc = args.get_arg<int>("kc");
    int mr = args.get_arg<int>("mr");
    int nr = args.get_arg<int>("nr");
    int l1_size = args.get_arg<int>("l1_size");
    int l2_size = args.get_arg<int>("l2_size");
    int l3_size = args.get_arg<int>("l3_size");
    int cacheline_size = args.get_arg<int>("cacheline_size");
    int page_size = args.get_arg<int>("page_size");
    int tlb_entry_l1d = args.get_arg<int>("tlb_entry_l1d");

    std::vector<int> affinity;
    affinity.push_back(cpu);
    set_current_affinity(affinity); // TODO: need disable intel HT
    std::vector<int> current_aff;
    get_current_affinity(current_aff);

    // construct the ctx
    gemm_context_t gemm_ctx;
    gemm_ctx.m  = m;
    gemm_ctx.n  = n;
    gemm_ctx.k  = k;
    gemm_ctx.layout  = layout;
    gemm_ctx.trans_a = trans_a;
    gemm_ctx.trans_b = trans_b;
    gemm_ctx.lda     = lda;
    gemm_ctx.ldb     = ldb;
    gemm_ctx.ldc     = ldc;
    gemm_ctx.alpha   = alpha;
    gemm_ctx.beta    = beta;
    gemm_ctx.alignment = align;

    gemm_ctx.mc = mc;
    gemm_ctx.nc = nc;
    gemm_ctx.kc = kc;
    gemm_ctx.mr = mr;
    gemm_ctx.nr = nr;

    gemm_ctx.cpu_list   = current_aff; // TODO: multiple thread
    gemm_ctx.l1_size    = l1_size;
    gemm_ctx.l2_size    = l2_size;
    gemm_ctx.l3_size    = l3_size;

    gemm_ctx.tlb_entry_l1d  = tlb_entry_l1d;
    gemm_ctx.cacheline_size = cacheline_size;
    gemm_ctx.page_size      = page_size;

    gemm_ctx.frequency = freq;

    //int current_cpu = get_current_cpu();
    //printf("current runing on cpu %d\n", current_cpu);

    // force single thread openblas
    //if(!no_ref)
    openblas_set_num_threads(1);

    gemm_bench<float> gb;
    gb.run(&gemm_ctx, valid, no_ref, one_shot);

    return 0;
}
