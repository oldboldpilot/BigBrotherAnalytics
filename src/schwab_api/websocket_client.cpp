#include "schwab_client.hpp"

namespace bigbrother::schwab {

class StreamingClient::Impl {
public:
    explicit Impl(std::shared_ptr<TokenManager>) {}
};

StreamingClient::~StreamingClient() = default;

} // namespace bigbrother::schwab
