#include "schwab_client.hpp"

namespace bigbrother::schwab {

class TradingClient::Impl {
public:
    explicit Impl(std::shared_ptr<TokenManager>) {}
};

TradingClient::~TradingClient() = default;

} // namespace bigbrother::schwab
