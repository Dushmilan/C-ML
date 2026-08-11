// autograd.h
typedef struct OpNode {
    struct OpNode** inputs;    // Array of input nodes
    size_t n_inputs;
    void (*backward)(struct OpNode*);
    Tensor* grad;
    Tensor** saved;            // Array of saved tensors for backward
    size_t n_saved;
} OpNode;

Tensor* tensor_create(size_t* shape, size_t ndim);
void tensor_backward(Tensor* tensor);  // Triggers full backprop