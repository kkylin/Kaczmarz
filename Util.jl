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

end#module
