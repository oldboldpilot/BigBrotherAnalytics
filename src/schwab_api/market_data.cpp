#include "schwab_client.hpp"

namespace bigbrother::schwab {

class MarketDataClient::Impl {
public:
    explicit Impl(std::shared_ptr<TokenManager>) {}
};

MarketDataClient::~MarketDataClient() = default;

} // namespace bigbrother::schwab
