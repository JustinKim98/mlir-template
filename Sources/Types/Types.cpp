//===- Types.cpp - Standalone dialect types ---------------------*- C++ -*-===//
//
// This file shows example of how types can be defined in MLIR
//
//===----------------------------------------------------------------------===//

#include "Types/Types.hpp"

namespace mlir {
namespace detail {
struct SampleTypeStorage : public TypeStorage {
  SampleTypeStorage(mlir::Type memberTy, std::int64_t intParam);

  // Key type for identifying type storage
  using KeyTy = std::tuple<mlir::Type, std::int64_t>;

  // Used to compare two types. If key equals, two types are considered as same
  // type
  bool operator==(const KeyTy &key) const {
    return key == std::tie(m_memberTy, m_param);
  }

  // Hash key used for storing type to TypeUniquer
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  // Gets static key from the key elements
  static KeyTy getKey(const mlir::Type memberTy, const std::int64_t intParam) {
    return std::make_tuple(memberTy, intParam);
  }

  // Returns key of current type
  KeyTy getAsKey() const { return std::make_tuple(m_memberTy, m_param); }

  // Constructs type with given key and allocator
  static SampleTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<SampleTypeStorage>()) SampleTypeStorage(
        std::get<mlir::Type>(key), std::get<std::int64_t>(key));
  }

private:
  mlir::Type m_memberTy;
  std::int64_t m_param;
};
} // namespace detail

SampleType SampleType::get(const mlir::Type memberTy,
                           const std::int64_t intParam) {
  return Base::get(memberTy.getContext(), memberTy, intParam);
}

mlir::Type SampleType::getMemberTy() const {
  return std::get<mlir::Type>(getImpl()->getAsKey());
}

std::int64_t SampleType::getIntParam() const {
  return std::get<std::int64_t>(getImpl()->getAsKey());
}

// `sample_type` `<` <memberTy> `, ` <intParam> `>`
SampleType SampleType::parseType(DialectAsmParser &parser) {
  if (parser.parseKeyword("sample_type"))
    return nullptr;

  mlir::Type memberTy = nullptr;
  std::int64_t intParam = 0;
  if (parser.parseLess() || parser.parseType(memberTy) || parser.parseComma() ||
      parser.parseInteger(intParam) || parser.parseGreater())
    return nullptr;

  return SampleType::get(memberTy, intParam);
}

void SampleType::printType(SampleType type, DialectAsmPrinter &printer) const {
  printer << "sample_type" << "<" << type.getMemberTy() << ", "
          << type.getIntParam() << ">";
}

} // namespace mlir
