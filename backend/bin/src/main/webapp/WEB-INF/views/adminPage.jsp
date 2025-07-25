<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ taglib uri="http://java.sun.com/jsp/jstl/fmt" prefix="fmt" %>
<%@ page contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

</head>
<body>
	<h1>전체 회원 정보</h1>
	
	<div>
		<button onclick="location.href='insertUser'" class="btn">회원 추가</button>
		<button onclick="location.href='insertFarm'" class="btn">(QC_Farm 테이블 정보) 추가</button>
	</div>
	<table>
		<tr>
			<th>번호</th>
			<th>이름</th>
			<th>아이디(휴대폰번호)</th>
			<th>대표농장이름</th>
			<th>대표농장주소</th>
			<th>대표농장번호(회사번호)</th>
			<th>가입 날짜</th>
		</tr>
		<c:forEach var="farm" items="${farmList}">
			<tr>
				<td>${farm.farmIdx}</td>
				<td>
				    <a href="/admin/edit?userPhone=${farm.userPhone}">
				        ${farm.userName}
				    </a>
				</td>
				<td>${farm.userPhone}</td>
				<td>${farm.farmName}</td>
				<td>${farm.farmAddr}</td>
				<td>${farm.farmPhone}</td>
				<td>${farm.joinedAt}</td>
			</tr>
		</c:forEach>
		<c:if test="${empty farmList}">
			<tr>
				<td colspan="7">검색 결과가 없습니다.</td>
			</tr>
		</c:if>
	</table>
	

	
	<form action="admin" method="get">
	    <select name="searchField">
	        <option value="user_name" ${searchField == 'user_name' ? 'selected' : ''}>회원 이름</option>
	        <option value="farm_name" ${searchField == 'farm_name' ? 'selected' : ''}>농장 주소</option>
	    </select>
	    <input type="text" name="keyword" placeholder="검색어 입력" value="${keyword}" required>
	    <input type="submit" value="검색">
	</form>
		

	<!-- 페이징처리 -->
	<c:forEach var="item" items="${list}">
	    <!-- 글 출력 -->
	    <div>
	        <h3>${item.title}</h3>
	        <p>${item.content}</p>
	        <!-- 필요한 데이터 출력 -->
	    </div>
	</c:forEach>
	
	<div class="pagination">
    <c:if test="${page > 1}">
        <a href="admin?page=${page - 1}&keyword=${keyword}">이전</a>
    </c:if>

    <c:forEach begin="1" end="${totalPage}" var="i">
        <c:choose>
            <c:when test="${i == page}">
                <b>${i}</b>
            </c:when>
            <c:otherwise>
                <a href="admin?page=${i}&keyword=${keyword}">${i}</a>
            </c:otherwise>
        </c:choose>
    </c:forEach>

    <c:if test="${page < totalPage}">
        <a href="admin?page=${page + 1}&keyword=${keyword}">다음</a>
    </c:if>
</div>
	

</body>
</html>
