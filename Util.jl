module Util

export TimeReporter

function TimeReporter(maxcount;
                      tag="",
                      period=60.0, # sec
                      )

    let t0    = time(),
        tlast = 0.0,
        count = 0,

        function()
            count = count + 1
            dt = time() - t0
            if dt >= tlast + delay
                tlast = dt
                println("#", tag, ": ", count, "/",
                        maxcount,
                        " steps took ",
                        nicedate(dt), "; eta ",
                        nicedate((dt/count)*(maxcount-count)))
                flush(stdout)
            end
        end
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
