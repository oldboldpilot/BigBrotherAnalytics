#include "schwab_client.hpp"

namespace bigbrother::schwab {

class AccountClient::Impl {
public:
    explicit Impl(std::shared_ptr<TokenManager>) {}
};

AccountClient::~AccountClient() = default;

} // namespace bigbrother::schwab
