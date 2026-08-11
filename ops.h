// ops.h
Tensor* matmul(const Tensor* A, const Tensor* B);
Tensor* add(const Tensor* A, const Tensor* B);
Tensor* relu(const Tensor* A);
Tensor* softmax(const Tensor* A, int axis);
Tensor* cross_entropy_loss(const Tensor* logits, const Tensor* targets);