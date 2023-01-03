module Helper

    export rowwise

    function rowwise(f, arr)
        [f(arr[i,:]) for i in 1:size(arr)[1]]
    end
end