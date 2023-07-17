def plot(results, y_axis='f_value',x_axis='iterations', time_offset=0.1, xlim_max=None, ylim_max=None):
    from matplotlib.colors import rgb2hex
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))

    # cmap = plt.get_cmap("Pastel1",256)
    cmap = plt.get_cmap("Set2",256)
    palette = cmap.colors
    palette = pd.DataFrame({'color': [rgb2hex(x) for x in palette]}).drop_duplicates()
    palette = list(palette['color'].values)

    for i, result in enumerate(results):
        # print('_'*50)
        col_name = result['optimation']
        # print(col_name)
        if result['use_backtracking'] == True:
            col_name += '_LS'
        parameter = result['parameter']
        for key, value in parameter.items():
            col_name += '_' + key + '=' + value

        f_value = result['result'][y_axis]
        if x_axis == 'iterations':
            df = pd.DataFrame({x_axis: range(len(f_value))})
        else:
            df = pd.DataFrame({x_axis: result['result'][x_axis]})
        df[col_name] = f_value
        plt.plot(x_axis, col_name, data=df,color=palette[i],linewidth=2,marker='o')

    if xlim_max != None and x_axis=='iterations':
        plt.xlim(-5,xlim_max)
    if xlim_max != None and x_axis=='time':
        plt.xlim(time_offset,xlim_max)
    if ylim_max != None:
        plt.ylim(0,ylim_max)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()
    
    
def print_result(config):
    alg_lst = []
    params = []
    t_lst = []
    f_lst = []
    time_lst = []
    for i, con in enumerate(config):
        # print('_'*50)
        optimation = con['optimation']
        use_backtrack = con['use_backtracking']
        if use_backtrack:
            alg_lst.append(optimation + "_" + "LS")
        else:
            alg_lst.append(optimation)
        param = str(i) + '_' + '_'.join([key+'='+value for key, value in con['parameter'].items()])
        result = con['result']
        f_value = result['f_value'][-1]
        t = len(result['f_value'])
        t_value = result['time'][-1]
        params.append(param)
        t_lst.append(t)
        f_lst.append(f_value)
        time_lst.append(t_value)
        # print(param)
        # print(f'finish after {t} iterations')
        # print('final f_value',round(f_value,6))
        # print('total time', round(t_value,6))
    df = pd.DataFrame({'algorithm': alg_lst, 'parameters': params, 'stop_iteration': t_lst, 'final_fvalue': f_lst, 'total_time': time_lst})
    df = df.set_index(['algorithm','parameters'])
    # df = df.sort_values(by=['final_fvalue','stop_iteration','total_time'])
    def highlight_min(s, props=''):
        return np.where(s == np.nanmin(s.values), props, '')
    display(df.style.apply(highlight_min, props='color:white;background-color:#2B61A1', axis=0))
    return df.reset_index()