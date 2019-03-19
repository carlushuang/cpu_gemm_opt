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
#include <fstream>

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

    // ugly! not c buffer is not owned
    bench_result & operator= (const bench_result & rhs){
        loops = rhs.loops;
        gflops = rhs.gflops;
        time_ms = rhs.time_ms;
        perf = rhs.perf;
        return *this;
    }
};

#define LOOPS 6
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
        int l_warmup = this->loop_warmup;
        int l_loop = this->loops;
        // some speed up and stability modify
        if(ctx->m < 600){
            l_warmup = 5;
            l_loop = 12;
        }
        else if(ctx->m < 1800){
            l_warmup = 4; 
            l_loop = 8;
        }
        else if(ctx->m > 3000){
            l_warmup = 2;
        }
        else if(ctx->m > 6000){
            l_warmup = 2;
            l_loop = 4;
        }
        for(i=0;i<l_warmup;i++){
            gemm_func(ctx->layout,ctx->trans_a,ctx->trans_b,
                ctx->m,ctx->n,ctx->k,
                ctx->alpha,
                A->data,ctx->lda,
                B->data,ctx->ldb,
                ctx->beta,
                c_out->data, ctx->ldc, ctx);
        }
        double start_time = current_sec();
        for(i=0;i<l_loop;i++){
            gemm_func(ctx->layout,ctx->trans_a,ctx->trans_b,
                ctx->m,ctx->n,ctx->k,
                ctx->alpha,
                A->data,ctx->lda,
                B->data,ctx->ldb,
                ctx->beta,
                c_out->data, ctx->ldc, ctx);
        }
        double cost_per_loop = (current_sec()-start_time) / l_loop;
        unsigned long long flop = sgemm_flop(ctx->m,ctx->n,ctx->k,ctx->alpha,ctx->beta);
        double gflops = (double)flop/(cost_per_loop *1e9);
        double gflops_theory = peak_gflops_t<T>()(ctx->frequency);
        delete c_out;
        //return std::move(bench_result(LOOPS, gflops, cost_per_loop*1e3, gflops/gflops_theory*100, nullptr));
        return bench_result<T>(l_loop, gflops, cost_per_loop*1e3, gflops/gflops_theory*100, nullptr);
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

        void serialize(std::ostream & os){
            os<<m<<"-"<<n<<"-"<<k<<"-";
        }
        void serialize(std::string & str){
            std::ostringstream oss;
            serialize(oss);
            str = oss.str();
        }
        void deserialize(std::istream & is){
            char _d;
            is>>m>>_d>>n>>_d>>k;
        }
        void deserialize(std::string & str){
            std::istringstream iss;
            iss.str(str);
            deserialize(iss);
        }
    };
    struct blocking_param{
        size_t mc;
        size_t nc;
        size_t kc;
        size_t mr;  // TODO
        size_t nr;

        void serialize(std::ostream & os){
            os<<mc<<"|"<<nc<<"|"<<kc<<"|"<<mr<<"|"<<nr;
        }
        void serialize(std::string & str){
            std::ostringstream oss;
            serialize(oss);
            str = oss.str();
        }
        void deserialize(std::istream & is){
            unsigned char _d;
            is>>mc>>_d>>nc>>_d>>kc>>_d>>mr>>_d>>nr;
        }
        void deserialize(std::string & str){
            std::istringstream iss;
            iss.str(str);
            deserialize(iss);
        }
    };

    std::unordered_map<std::string, std::string> tuned_blocking_map;
    void serialize_map(const std::unordered_map<std::string, std::string> & map,
            const std::string file_name)
    {
        std::ofstream outfile;
        outfile.open(file_name, std::ios_base::app);
        for(auto & item : map){
            outfile<<item.first<<":"<<item.second<<std::endl;
        }
    }
    void serialize_pair(const std::string & key, const std::string & value,
            const std::string file_name)
    {
        std::ofstream outfile;
        outfile.open(file_name, std::ios_base::app);
        outfile<<key<<":"<<value<<std::endl;
    }
    void deserialize_map(std::unordered_map<std::string, std::string> & map,
            const std::string file_name)
    {
        std::ifstream infile(file_name);
        if(!infile.good())
            return ;
        std::string line;
        while (std::getline(infile, line))
        {
            size_t col_pos = line.find_first_of(':');
            size_t col_pos_next = line.find_first_of(':' , col_pos+1);
            std::string map_key = line.substr(0, col_pos);
            std::string map_value;
            if(col_pos_next == std::string::npos){
                map_value = line.substr(col_pos+1);
            }else{
                map_value = line.substr(col_pos+1, col_pos_next-col_pos-1);
            }
            map[map_key] = map_value;
        }
    }
    void update_tuned_param(const std::unordered_map<std::string, std::string> & map,
        gemm_context_t *ctx, const blocking_param & default_bp){
        std::string key;
        ctx->serialize(key);
        ctx->cur_use_tuned = false;
        if(map.count(key)){
            std::string value = map.at(key);
            blocking_param bp;
            bp.deserialize(value);

            ctx->mc = bp.mc;
            ctx->nc = bp.nc;
            ctx->kc = bp.kc;
            //std::cout<<"    update param for "<<key<<std::endl;
            ctx->cur_use_tuned = true;
        }else{
            //std::cout<<"    no param update for "<<key<<std::endl;
            ctx->mc = default_bp.mc;
            ctx->nc = default_bp.nc;
            ctx->kc = default_bp.kc;
        }
    }

    std::string get_tuned_db_filename(){
        // TODO: better file name
        std::string fn;
        std::string gemm_name;
        if(sizeof(T) == 4)
            gemm_name = "sgemm";
        else if(sizeof(T) == 8)
            gemm_name = "dgemm";
        fn = gemm_name + "_tuned.db";
        return fn;
    }

    bool next_config(config * cfg){
        static int ITER_START = 48;
        static int ITER_STEP = 48;
        static int ITER_END = 9218;
        //static int ITER_END = 1024;
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
            else if(current_iter < 2048)
                step = ITER_STEP * 4;
            else if(current_iter < 4096)
                step = ITER_STEP * 6;
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
    inline size_t req_l1(size_t mc, size_t nc, size_t kc, size_t mr, size_t nr, size_t dsize){
        return (mr*kc+nr*kc+mr*nr+nr+mr*nr) * dsize;
    }
    inline size_t req_l2(size_t mc, size_t nc, size_t kc, size_t mr, size_t nr, size_t dsize){
        return (nc*kc+mr*kc+mr*nc+mr*kc+mr*nc) * dsize;
    }
    inline size_t req_l3(size_t mc, size_t nc, size_t kc, size_t mr, size_t nr, size_t dsize){
        return (mc*kc + nc*kc + mc*nc + nc*kc + mc*nc) * dsize;
    }
    inline size_t req_l1d_tlb(size_t mc, size_t nc, size_t kc, size_t mr, size_t nr, size_t dsize,
        size_t page_size)
    {
        size_t ta = CEIL(mr*kc*dsize, page_size)+1;
        size_t tb = CEIL(nr*kc*dsize, page_size)+1;
        size_t tc = mr;
        return ta+2*tb+tc;
    }

    struct stepping_t{
        size_t start = 0;
        size_t step = 0;
        size_t end = 0;
        stepping_t(){}
        stepping_t(size_t start_, size_t step_, size_t end_){
            start=start_;
            step=step_;
            end=end_;
        }
    };
    void get_current_stepping_t(const gemm_context_t *ctx, 
        stepping_t & ms, stepping_t & ns, stepping_t & ks)
    {
        size_t m, n, k;
        m = ctx->m;
        n = ctx->n;
        k = ctx->k;
        size_t mr, nr;
        mr = ctx->mr;
        nr = ctx->nr;

        assert( (m%2 == 0) && (m%4==0) && (n%2 == 0) && (n%4==0) && (k%2==0));
        // TODO: better solution
        if(m==48 && n==48 && k==48){
            ms = stepping_t(12, 6,  320);
            ns = stepping_t(16, 16, 320);
            ks = stepping_t(16, 16, 280);
            return ;
        }
        else if(m==96 && n==96 && k==96){
            ms = stepping_t(36, 12, 640);
            ns = stepping_t(48, 32, 640);
            ks = stepping_t(64, 16, 400);
            return ;
        }
        else if(m==144 && n==144 && k==144){
            ms = stepping_t(48, 24, 640);
            ns = stepping_t(64, 48, 640);
            ks = stepping_t(64, 32, 600);
            return ;
        }
        else if(m < 512 && n < 512 && k < 512){
            ms = stepping_t(CEIL_WRAP(m/4,mr), 24, m*3);
            ns = stepping_t(CEIL_WRAP(n/4,nr), 48, n*3);
            ks = stepping_t(CEIL_WRAP(k/4,16), 64, k*3);
            return ;
        }
        else if(m < 1024 && n < 1024 && k < 1024){
            ms = stepping_t(CEIL_WRAP(m/4,mr), 24, m*2);
            ns = stepping_t(CEIL_WRAP(n/4,nr), 48, n*2);
            ks = stepping_t(128, 64, k*2);
            return ;
        }
        else if(m < 2048 && n < 2048 && k < 2048){
            ms = stepping_t(504 , 48, m*2);
            ns = stepping_t(384, 64, n*2);
            ks = stepping_t(160, 96, k*2);
            return ;
        }
        else if(m < 4096 && n < 4096 && k < 4096){
            ms = stepping_t(480 , 48, m*2);
            ns = stepping_t(384, 64, n*2);
            ks = stepping_t(160, 96, 4096);
            return ;
        }

        ms = stepping_t(480 , 48, m);
        ns = stepping_t(384, 64, n);
        ks = stepping_t(160, 96, 4096);
        return ;
    }
    bool next_blocking_param(const gemm_context_t *ctx, blocking_param * bp){

        static size_t mm = 0;
        static size_t nn = 0;
        static size_t kk = 0;
        static stepping_t ms;
        static stepping_t ns;
        static stepping_t ks;

        static size_t cur_mc = ms.start;
        static size_t cur_nc = ns.start;
        static size_t cur_kc = ks.start;
        static size_t cur_mr = 6;
        static size_t cur_nr = 16;

        if(mm != ctx->m || nn != ctx->n || kk != ctx->k){
            get_current_stepping_t(ctx, ms, ns, ks);
            mm = ctx->m; nn = ctx->n; kk = ctx->k;

            cur_mc = ms.start;
            cur_nc = ns.start;
            cur_kc = ks.start;
        }

        static bool need_stop = false;
        if(need_stop){
            need_stop = false;
            return false;
        }

        bp->mc = cur_mc;
        bp->nc = cur_nc;
        bp->kc = cur_kc;
        bp->mr = cur_mr;     // TODO: only this kernel
        bp->nr = cur_nr;

        size_t l1_size = ctx->l1_size;
        size_t l2_size = ctx->l2_size;
        size_t l3_size = ctx->l3_size;
        size_t tlb_entry_l1d = ctx->tlb_entry_l1d;
        size_t page_size = ctx->page_size;

        auto valid_req_func = [&](){
            bool valid_l1 = req_l1(cur_mc, cur_nc, cur_kc, cur_mr, cur_nr, sizeof(T)) < l1_size;
            bool valid_l2 = req_l2(cur_mc, cur_nc, cur_kc, cur_mr, cur_nr, sizeof(T)) < l2_size;
            bool valid_l3 = req_l3(cur_mc, cur_nc, cur_kc, cur_mr, cur_nr, sizeof(T)) < l3_size;
            bool valid_l1d_tlb = req_l1d_tlb(cur_mc, cur_nc, cur_kc, cur_mr, cur_nr, sizeof(T), page_size) < tlb_entry_l1d;
            return valid_l1 && valid_l2 && valid_l3 && valid_l1d_tlb;
        };

        cur_mc += ms.step;
        if(cur_mc > ms.end || !valid_req_func()){
            cur_mc = ms.start;
            cur_nc += ns.step;
            if(cur_nc > ns.end || !valid_req_func()){
                cur_nc = ns.start;
                cur_kc += ks.step;
                if(cur_kc > ks.end || !valid_req_func()){
                    cur_kc = ks.start;
                    need_stop = true;
                }
            }
        }

        return true;
    }
    std::string cpu_list_to_str (const std::vector<int> & cpu_list_){
        std::string str;
        for(int i=0;i<cpu_list_.size();i++){
            str += std::to_string(cpu_list_[i]);
            if(i != (cpu_list_.size()-1) )
                str += ",";
        }
        return str;
    }
    std::string byte_2_str (size_t bytes){
        std::string str;
        if(bytes < 1024){
            str += std::to_string(bytes);
            str += "B";
        }else if (bytes < (1024*1024)){
            size_t b;
            size_t k;
            b = bytes % (1024);
            k = bytes / 1024;
            str += std::to_string(k);
            if(b != 0){
                str += ".";
                str += std::to_string(b*100/1024);  // .2f
            }
            str += "K";
        }else{
            size_t m;
            size_t k;

            m = bytes / (1024*1024);
            k = bytes % (1024*1024);
            str += std::to_string(m);
            if(k != 0){
                str += ".";
                str += std::to_string(k*100/(1024*1024));
            }
            str += "M";
        }
        return str;
    }
    void dump_ctx (const gemm_context_t *ctx, int dump_level = 1){
        size_t mc, nc, kc, mr, nr;
        size_t l1_size, l2_size, l3_size, tlb_entry_l1d, page_size;
        mc = ctx->mc;
        nc = ctx->nc;
        kc = ctx->kc;
        mr = ctx->mr;
        nr = ctx->nr;
        l1_size = ctx->l1_size;
        l2_size = ctx->l2_size;
        l3_size = ctx->l3_size;
        tlb_entry_l1d = ctx->tlb_entry_l1d;
        page_size = ctx->page_size;

        std::string cpu_list_str = cpu_list_to_str(ctx->cpu_list);
        printf("cpu:%s, freq: %.1fMHz, theoritical: %.3f gflops (avx256,fmadd)\n",
                        cpu_list_str.c_str(), ctx->frequency, peak_gflops_t<T>()(ctx->frequency));

        std::string l1_size_str = byte_2_str(l1_size);
        std::string l2_size_str = byte_2_str(l2_size);
        std::string l3_size_str = byte_2_str(l3_size);
        printf("l1_size:%s, l2_size:%s, l3_size:%s, page_size:%lu, tlb_entry_l1d:%lu\n",
                        l1_size_str.c_str(), l2_size_str.c_str(), l3_size_str.c_str(), page_size, tlb_entry_l1d);
        if(dump_level < 1)
            return ;
        printf("MC:%lu, NC:%lu, KC:%lu, MR:%lu, NR:%lu\n",
                        mc, nc, kc, mr, nr);
        //printf("layout:%s, trans_a:%s, trans_b:%s\n",
        //                to_layout_str(ctx->layout), to_trans_str(ctx->trans_a), to_trans_str(ctx->trans_b));
        printf("Considerations:\n");
        if(ctx->layout == LAYOUT_ROW_MAJOR){
            size_t lhs, rhs;

            lhs = req_l1(mc, nc, kc, mr, nr, 1); // mr*kc+nr*kc+mr*nr+nr+mr*nr;
            rhs = l1_size/sizeof(T);
            printf(" L1: MR*KC+NR*KC+MR*NR+NR+MR*NR < L1_size/d_size, lhs:%lu, rhs:%lu, match?%s\n",
                    lhs, rhs, ((lhs<rhs)?"yes":"no"));
            
            lhs = req_l2(mc, nc, kc, mr, nr, 1); //nc*kc+mr*kc+mr*nc+mr*kc+mr*nc;
            rhs = l2_size/sizeof(T);
            printf(" L2: NC*KC+MR*KC+MR*NC+MR*KC+MR*NC < L2_size/d_size, lhs:%lu, rhs:%lu, match?%s\n",
                    lhs, rhs, ((lhs<rhs)?"yes":"no"));

            lhs = req_l3(mc, nc, kc, mr, nr, 1); // mc*kc + nc*kc + mc*nc + nc*kc + mc*nc;
            rhs = l3_size/sizeof(T);
            printf(" L3: MC*KC+NC*KC+MC*NC+NC*KC+MC*NC < L3_size/d_size, lhs:%lu, rhs:%lu, match?%s\n",
                    lhs, rhs, ((lhs<rhs)?"yes":"no"));

            printf(" L1D TLB:\n");

            size_t ta, tb, tc;
            ta = CEIL(mr*kc*sizeof(T), page_size)+1;
            tb = CEIL(nr*kc*sizeof(T), page_size)+1;
            tc = mr;
            printf("  TA:CEIL(MR*KC*d_size/PAGE_SIZE)+1, %lu\n", ta);
            printf("  TB:CEIL(NR*KC*d_size/PAGE_SIZE)+1, %lu\n", tb);
            printf("  TC:up to MR, %lu\n", tc);

            lhs = ta+2*tb+tc;
            rhs = tlb_entry_l1d;
            printf("  TA+2*(TB)+TC < T_entry_total, lhs:%lu, rhs:%lu, match?%s\n",
                        lhs, rhs, ((lhs<rhs)?"yes":"no"));
        }else{

        }
    }

    void tune(gemm_context_t *ctx){
        // TODO: here tune only for square matrix
        auto summary_func = [&](gemm_context_t *ctx, bench_result<T> * ref, blocking_param * bp){
            size_t mc = bp->mc;
            size_t nc = bp->nc;
            size_t kc = bp->kc;
            size_t mr = bp->mr;
            size_t nr = bp->nr;
            size_t page_size = ctx->page_size;

            std::string l1 = byte_2_str(req_l1(mc, nc, kc, mr, nr, sizeof(T)));
            std::string l2 = byte_2_str(req_l2(mc, nc, kc, mr, nr, sizeof(T)));
            std::string l3 = byte_2_str(req_l3(mc, nc, kc, mr, nr, sizeof(T)));
            std::string l1tlb = std::to_string(req_l1d_tlb(mc, nc, kc, mr, nr, sizeof(T), page_size));

            printf(" %4lu %4lu %4lu  %.1f  %.1f"
                "  %4lu %4lu %4lu %3lu %3lu %7.3f %7.2f(%2.2f)  %s/%s/%s/%s ",
                ctx->m,ctx->n,ctx->k,ctx->alpha,ctx->beta,
                bp->mc, bp->nc, bp->kc, bp->mr, bp->nr,
                ref->time_ms, ref->gflops ,ref->perf,
                l1.c_str(), l2.c_str(), l3.c_str(), l1tlb.c_str());
            printf("\n");
        };
        dump_ctx(ctx, 0);
        printf("    M    N    K alpha beta   mc   nc   kc  mr  nr  best(ms)  gflops(%%)    req(l1/l2/l3/l1dtlb)\n");
        config cfg;
        blocking_param bp;

        std::string db_fn = get_tuned_db_filename();
        while( next_config(&cfg) ){
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

            double time_ms = 99999999.99f;
            blocking_param  best_bp;
            bench_result<T> best_result;
            //printf("---- start run\n");
            while( next_blocking_param(ctx, &bp) ){

                ctx->mc      = bp.mc;
                ctx->nc      = bp.nc;
                ctx->kc      = bp.kc;
                ctx->mr      = bp.mr;
                ctx->nr      = bp.nr;
                if( !bp.mc || !bp.nc || !bp.kc || (bp.mc%bp.mr) || (bp.nc%bp.nr) ){
                    printf("  m:%lu, n:%lu, k:%lu, mc:%lu, nc:%lu, kc:%lu, mr:%lu, nr:%lu\n",
                       ctx->m, ctx->n, ctx->k, bp.mc, bp.nc, bp.kc, bp.mr, bp.nr);
                    assert(0);
                }
                //printf("  m:%lu, n:%lu, k:%lu, mc:%lu, nc:%lu, kc:%lu, mr:%lu, nr:%lu\n",
                //    ctx->m, ctx->n, ctx->k, bp.mc, bp.nc, bp.kc, bp.mr, bp.nr);

                gemm_problem_t<T> gemm_prob(ctx);
                bench_result<T> rtn_opt = gemm_prob.run_single_case(cblas_sgemm_opt, false);
                if(rtn_opt.time_ms < time_ms){
                    time_ms = rtn_opt.time_ms;
                    best_bp = bp;
                    best_result = rtn_opt;
                }
            }
            summary_func(ctx, &best_result, &best_bp);
            std::string map_key, map_value;

            ctx->serialize(map_key);
            best_bp.serialize(map_value);

            serialize_pair(map_key,map_value,db_fn);
            tuned_blocking_map[map_key] = map_value;
        }
    }
    //void run(std::vector<int> cpu_list, double freq, bool validate_only, bool no_ref, gemm_problem_t * single_problem = nullptr){
    void run(gemm_context_t *ctx, bool validate_only, bool no_ref, bool one_shot, bool use_tuned){

        auto summary_func = [&](gemm_problem_t<T> * prob, bench_result<T> * r_ref, bench_result<T> * r_opt){
            printf(" %4lu %4lu %4lu  %.1f  %.1f "
                    "  %4lu %4lu %4lu %3lu %3lu %6.2f(%2.2f) %6.2f(%2.2f)",
                prob->ctx->m,prob->ctx->n,prob->ctx->k,prob->ctx->alpha,prob->ctx->beta,
                prob->ctx->mc, prob->ctx->nc, prob->ctx->kc, prob->ctx->mr, prob->ctx->nr,
                r_opt->gflops,r_opt->perf,r_ref?(r_ref->gflops):0,r_ref?(r_ref->perf):0);
            if(prob->ctx->cur_use_tuned)
                printf("  [t]");
            else
                printf("  [*]");
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
        if(use_tuned)
            deserialize_map(tuned_blocking_map, get_tuned_db_filename());

        dump_ctx(ctx);
        assert( ((ctx->mc % ctx->mr) == 0) && ((ctx->nc % ctx->nr) == 0) &&
                    "MC%%MR, NC%%NR must be zero\n");
        
        blocking_param default_bp;
        default_bp.mc = ctx->mc;
        default_bp.nc = ctx->nc;
        default_bp.kc = ctx->kc;
        default_bp.mr = ctx->mr;
        default_bp.nr = ctx->nr;

        //printf("require: L1:%.1fKB(KC*NR*4), L2:%.1fKB(KC*MC*4), L3:%.1fKB(KC*NC*4)\n", req_l1()/1024.0, req_l2()/1024.0, req_l3()/1024.0);
        printf("    M    N    K alpha beta   mc    nc   kc  mr  nr   gflops(%%)   gflops_ref(%%)\n");

        while(1){
            if(one_shot){
                if(use_tuned)
                    update_tuned_param(tuned_blocking_map, ctx, default_bp);
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
                if(use_tuned)
                    update_tuned_param(tuned_blocking_map, ctx, default_bp);

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
    args.insert_arg("m", "M value of gemm, int", "576");
    args.insert_arg("n", "N value of gemm, int", "576");
    args.insert_arg("k", "K value of gemm, int", "576");
    args.insert_arg("lda", "leading dimension of a", "576");
    args.insert_arg("ldb", "leading dimension of b", "576");
    args.insert_arg("ldc", "leading dimension of c", "576");
    args.insert_arg("a", "ALPHA value of gemm, double", "1.0");
    args.insert_arg("b", "BETA value of gemm, double", "0");
    args.insert_arg("f", "CPU frequency, in MHz, double", "2600");

    args.insert_arg("tune", "tuning blocking params", "0");
    args.insert_arg("use_tuned", "use previously tuned db. if file not exist, ignore", "1");
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

    bool tune  = (args.get_arg<int>("tune")==1)?true:false;
    bool use_tuned  = (args.get_arg<int>("use_tuned")==1)?true:false;
#if 0
    if(args.used_arg("mc") || args.used_arg("nc") || args.used_arg("kc") 
        || args.used_arg("mr") || args.used_arg("nr"))
        use_tuned = false;
#endif
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
    if(tune){
        gb.tune(&gemm_ctx);
    }else
        gb.run(&gemm_ctx, valid, no_ref, one_shot, use_tuned);

    return 0;
}
