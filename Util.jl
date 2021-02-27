module Util

import Base.foreach

function foreach(f::Function, range, tag::AbstractString; delay=1.0)
    let t0::Float64 = time(),
        tlp = 0.0,
        count::Int64 = 0,
        total::Int64 = length(range)

        let imin = minimum(range),
            imax = maximum(range)

            for i in range
                f(i)
                count = count + 1
                dt = time() - t0
                if dt >= tlp + delay || i == imin || i == imax
                    tlp = dt
                    println("#", tag, ": ", count, "/",
                            total,
                            " steps took ",
                            nicedate(dt), "; eta ",
                            nicedate((dt/count)*(total-count)))
                    flush(stdout)
                end
            end
        end
        return time()-t0
    end
end

function nicedate(sec::Float64)
    if sec < 60
        sec = chop(sec)
        return "$sec sec"
    else
        min = floor(Int,sec / 60)
        sec = chop( sec - 60*min )
        if min < 60
            return "$min min $sec sec"
        else
            hrs = floor(Int,min / 60)
            min -= 60*hrs
            if hrs < 24
                return "$hrs hrs $min min $sec sec"
            else
                days = floor(Int,hrs / 24)
                hrs -= 24*days
                return "$days days $hrs hrs $min min $sec sec"
            end
        end
    end
end

function chop(t::Float64; n::Int64=3)
    floor(Int,t*10^n)/10^n
end

end#module
