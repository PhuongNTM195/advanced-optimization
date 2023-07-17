def backtracking_line_search_wolfe1(x_k, alpha_bar=1, ro=0.5, c=1e-4):
    # Tìm kiếm theo tia theo phương pháp backtracking
    # Khởi tạo các tham số alpha > 0, ro nằm trong đoạn (0,1), c nằm trong đoạn (0,1)
    alpha = alpha_bar
    # vector gradient tại x_k
    grad_k = grad_F(x_k)
    p_k = -grad_k
    # giá trị hàm f() tại x_k
    f_k = F(x_k)
    # gradient tại x_k * p_k
    grad_p_k = c*grad_k.T@p_k
    # cập nhật giá trị của function tại x_k
    f_new = F(x_k + alpha*p_k)
    while f_new > f_k + alpha*grad_p_k:
        alpha *= ro
        f_new = F(x_k + alpha*p_k)
    return alpha