from perfetto.trace_processor import TraceProcessor
import pandas as pd
import argparse

def dims_to_sz(row):

    # the function is kinda messy, for newer versions of Megatron-LM and TensorEngine the name and format of 
    # the GEMM C++ API might change, so keep this in mind if using this in the future

    if (row["name_parent"] == "aten::mm"): # aten (PyTorch C++ backend) argument format
        m = row["dims"]['Dims[0][0]']
        k = row["dims"]['Dims[0][1]']
        n = row["dims"]['Dims[1][1]']
    else: # TensorEngine argument format (https://hub.docker.com/layers/rocm/megatron-lm/v25.5_py312/images/sha256-4506f18ba188d24189c6b1f95130b425f52c528a543bb3f420351824edceadc2)
        a = row["dims"]['Dims[0][0]']
        b = row["dims"]['Dims[0][1]']

        c = row["dims"]['Dims[5][0]']
        d = row["dims"]['Dims[5][1]']

        m = row["dims"]["Dims[10][0]"]
        n = row["dims"]["Dims[10][1]"]
        if (m == c):
            k = d
        else:
            k = c
    return m,n,k

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        prog="shapes_finder",
        description="find gemm shapes in a perfetto trace"
    )
    parser.add_argument('-f', '--filename', required=True)      # option that takes a value

    args = parser.parse_args()

    tp = TraceProcessor(trace=args.filename)

    # we need the flows table, since we need to time hipBLASlt kernels,
    # which are launched from the thread using hipExtModuleLaunchKernel,
    # since the kernel does not contain any of its shape arguments in the trace
    # (it only has grid and block size)
    df_flows = tp.query('SELECT * FROM flow').as_pandas_dataframe() 
    df_slice = tp.query('SELECT * FROM slice').as_pandas_dataframe()
    df_args = tp.query('SELECT * FROM args').as_pandas_dataframe()

    df_shapes = df_args[df_args["key"].str.contains("Input Dims")].groupby("arg_set_id").agg(list)
    df_shapes["key"] = df_shapes["key"].apply(lambda k: [x[11:] for x in k])
    df_shapes["dims"] = df_shapes[["key", "int_value"]].apply(
        lambda row: dict(zip(row["key"], row["int_value"])), axis=1
    )
    sl_names = df_slice["name"].str

    # obtain the host GEMM function slice
    gemm_mask = (sl_names.contains("aten::mm") | (sl_names.contains("te_gemm") & ~sl_names.contains("PyCapsule")))
    df_gemms = df_slice[gemm_mask]
    df_gemms_with_shapes = pd.merge(df_gemms, df_shapes, on="arg_set_id", how="inner")

    merged = pd.merge(df_slice, df_gemms_with_shapes, left_on="parent_id", right_on="id_x", suffixes=["_child", "_parent"])
    launches = merged[merged["name_child"].str.contains("Launch")]

    partial = pd.merge(df_flows[["slice_in", "slice_out"]], launches, how="inner", left_on="slice_out", right_on="id")
    kernels = pd.merge(df_slice, partial, how="inner", left_on="id", right_on="slice_in", suffixes=["_kernel", "_caller"])

    kernels = kernels[kernels["name"].str.contains("Cijk")]

    cols_to_use = ["id_kernel", "name", "dur", "dims", "name_parent"]
    kernels = kernels[cols_to_use]
    kernels[["m", "n", "k"]] = kernels.apply(dims_to_sz, axis=1,result_type="expand")
    kernels["tflop/s"] = 2*kernels["m"]*kernels["k"]*kernels["n"] / kernels["dur"] * 1e-3

    kernels[["name", "dur", "name_parent", "m", "n", "k", "tflop/s"]].to_csv("gemm_shapes.csv")
