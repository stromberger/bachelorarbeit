module LFUCache

export LFU

mutable struct LFU{K, V} <: AbstractDict{K, V}
    dict::Dict{K, V}
    freq::Dict{K, Int}
    capacity::Int
    size::Int

    function LFU{K, V}(capacity::Int) where {K, V}
        new{K, V}(Dict{K, V}(), Dict{K, Int}(), capacity, 0)
    end
end

LFU(;maxsize=10::Int) = LFU{Any,Any}(maxsize)

Base.iterate(lfu::LFU) = iterate(lfu.dict)

function Base.setindex!(lfu::LFU{Any, Any}, value::Any, key::Any)
    if haskey(lfu.dict, key)
        lfu.dict[key] = value
        return value
    end

    if lfu.size == lfu.capacity
        min_key = sort(collect(lfu.freq), by=x->x[2])[1][1]
        delete!(lfu.dict, min_key)
        delete!(lfu.freq, min_key)
        lfu.size -= 1
    end

    lfu.dict[key] = value
    lfu.freq[key] = 1
    lfu.size += 1
    return value
end

function Base.getindex(lfu::LFU{Any, Any}, key::Any)
    if haskey(lfu.dict, key)
        lfu.freq[key] += 1
        return lfu.dict[key]
    end
    throw(KeyError(key))
end

function Base.keys(lfu::LFU{Any, Any})
    keys(lfu.dict)
end


end