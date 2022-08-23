#include "ddptensor/TypeDispatch.hpp"
#include "ddptensor/x.hpp"
#include "ddptensor/Deferred.hpp"
#include "ddptensor/Service.hpp"
#include "ddptensor/Factory.hpp"
#include "ddptensor/Registry.hpp"
#include "ddptensor/ddptensor.hpp"

namespace x {
    struct Service
    {
        using ptr_type = DPTensorBaseX::ptr_type;

        template<typename T>
        static ptr_type op(const std::shared_ptr<DPTensorX<T>> & a_ptr)
        {
            if(a_ptr->is_replicated()) return a_ptr;
            if(a_ptr->has_owner() && a_ptr->slice().size() == 1) {
                if(theTransceiver->rank() == a_ptr->owner()) {
                    a_ptr->_replica = *(xt::strided_view(a_ptr->xarray(), a_ptr->lslice()).begin());
                }
                theTransceiver->bcast(&a_ptr->_replica, sizeof(T), a_ptr->owner());
                a_ptr->set_owner(REPLICATED);
            } else {
                throw(std::runtime_error("Replication implemented for single element and single owner only."));
            }
            return a_ptr;
        }
    };
}

struct DeferredService : public Deferred
{
    enum Op : int {
        REPLICATE,
        DROP
    };

    id_type _a;
    Op _op;

    DeferredService() = default;
    DeferredService(Op op, const tensor_i::future_type & a)
        : _a(a.id()), _op(op)
    {}

    void run()
    {
#if 0
        switch(_op) {
        case REPLICATE: {
            const auto a = std::move(Registry::get(_a).get());
            set_value(std::move(TypeDispatch<x::Service>(a)));
            break;
        }
        case DROP:
            Registry::del(_a);
            break;
        default:
                throw(std::runtime_error("Unkown Service operation requested."));
        }
#endif
    }

    ::mlir::Value generate_mlir(::mlir::OpBuilder & builder, ::mlir::Location loc, jit::IdValueMap & ivm) override
    {
        switch(_op) {
        case DROP:
            if(auto e = ivm.find(_a); e != ivm.end()) {
                ivm.erase(e);
                // FIXME create delete op and return it
            }
            break;
        default:
            throw(std::runtime_error("Unkown Service operation requested."));
        }
        
        return {};
    }

    FactoryId factory() const
    {
        return F_SERVICE;
    }

    template<typename S>
    void serialize(S & ser)
    {
        ser.template value<sizeof(_a)>(_a);
        ser.template value<sizeof(_op)>(_op);
    }
};

ddptensor * Service::replicate(const ddptensor & a)
{
    return new ddptensor(defer<DeferredService>(DeferredService::REPLICATE, a.get()));
}

extern bool inited;

void Service::drop(const ddptensor & a)
{
    if(inited) {
        // if(is_spmd()) theTransceiver->barrier();
        defer<DeferredService>(DeferredService::DROP, a.get());
    }
}

FACTORY_INIT(DeferredService, F_SERVICE);
